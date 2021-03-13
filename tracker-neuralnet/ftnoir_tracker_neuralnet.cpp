/* Copyright (c) 2013-2015 Stanislaw Halik <sthalik@misaki.pl>
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 */

#include "ftnoir_tracker_neuralnet.h"
#include "compat/sleep.hpp"
#include "compat/math-imports.hpp"
#include "cv/init.hpp"
#include <eigen3/Eigen/src/Geometry/Quaternion.h>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include "compat/timer.hpp"
#include <omp.h>

#ifdef _MSC_VER
#   pragma warning(disable : 4702)
#endif

#include <QMutexLocker>
#include <QDebug>
#include <QFile>

#include <vector>
#include <tuple>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <iostream>

namespace
{

int enum_to_fps(int value)
{
    int fps;

    switch (value)
    {
    default: eval_once(qDebug() << "aruco: invalid fps enum value");
    [[fallthrough]];
    case fps_default:   fps = 0; break;
    case fps_30:        fps = 30; break;
    case fps_60:        fps = 60; break;
    case fps_75:        fps = 75; break;
    case fps_125:       fps = 125; break;
    case fps_200:       fps = 200; break;
    case fps_50:        fps = 50; break;
    case fps_100:       fps = 100; break;
    case fps_120:       fps = 120; break;
    case fps_300:       fps = 300; break;
    case fps_250:       fps = 250; break;
    }

    return fps;
}


#if _MSC_VER
std::wstring convert(const QString &s) { return s.toStdWString(); }
#else
std::string convert(const QString &s) { return s.toStdString(); }
#endif


float sigmoid(float x)
{
    return 1.f/(1.f + std::exp(-x));
}


template<class T>
cv::Rect_<T> squarize(const cv::Rect_<T> &r)
{
    cv::Point_<T> c{r.x + r.width/T(2), r.y + r.height/T(2)};
    const T sz = std::max(r.height, r.width);
    return {c.x - sz/T(2), c.y - sz/T(2), sz, sz};
}


int compute_padding(const cv::Rect &r, int w, int h)
{
    using std::max;
    return max({
        max(-r.x, 0),
        max(-r.y, 0),
        max(r.x+r.width-w, 0),
        max(r.y+r.height-h, 0)
    });
}


cv::Rect2f unnormalize(const cv::Rect2f &r, int h, int w)
{
    auto unnorm = [](float x) -> float { return 0.5*(x+1); };
    auto tl = r.tl();
    auto br = r.br();
    auto x0 = unnorm(tl.x)*w;
    auto y0 = unnorm(tl.y)*h;
    auto x1 = unnorm(br.x)*w;
    auto y1 = unnorm(br.y)*h;
    return {
        x0, y0, x1-x0, y1-y0
    };
}

cv::Point2f normalize(const cv::Point2f &p, int h, int w)
{
    return {
        p.x/w*2.f-1.f,
        p.y/h*2.f-1.f
    };
}

/* Computes correction due to head being off screen center.
    x, y: In screen space, i.e. in [-1,1]
    focal_length_x: In screen space
*/
Eigen::Quaternionf compute_rotation_correction(const cv::Point2f &p, float focal_length_x)
{
    return Eigen::Quaternionf::FromTwoVectors(
        Eigen::Vector3f::UnitX(),
        Eigen::Vector3f{focal_length_x, p.y, p.x});
}

template<class T>
T iou(const cv::Rect_<T> &a, const cv::Rect_<T> &b)
{
    auto i = a & b;
    return double{i.area()} / (a.area()+b.area()-i.area());
}


} // namespace


neuralnet_tracker::Localizer::Localizer(Ort::MemoryInfo &allocator_info, Ort::Session &&session) :
    session{std::move(session)},
    scaled_frame(input_img_height, input_img_width, CV_8U),
    input_mat(input_img_height, input_img_width, CV_32F)
{
    // Only works when input_mat does not reallocated memory ...which it should not.
    // Non-owning memory reference to input_mat?
    // Note: shape = (bach x channels x h x w)
    const std::int64_t input_shape[4] = { 1, 1, input_img_height, input_img_width };
    input_val = Ort::Value::CreateTensor<float>(allocator_info, input_mat.ptr<float>(0), input_mat.total(), input_shape, 4);

    const std::int64_t output_shape[2] = { 1, 5 };
    output_val = Ort::Value::CreateTensor<float>(allocator_info, results.data(), results.size(), output_shape, 2);
}


std::pair<float, cv::Rect2f> neuralnet_tracker::Localizer::run(
    const cv::Mat &frame)
{
    auto p = input_mat.ptr(0);

    cv::resize(frame, scaled_frame, { input_img_width, input_img_height }, 0, 0, cv::INTER_AREA);
    scaled_frame.convertTo(input_mat, CV_32F, 1./255., -0.5);

    assert (input_mat.ptr(0) == p);
    assert (!input_mat.empty() && input_mat.isContinuous());
    assert (input_mat.cols == input_img_width && input_mat.rows == input_img_height);

    const char* input_names[] = {"x"};
    const char* output_names[] = {"logit_box"};

    Timer t_; t_.start();

    const auto nt = omp_get_num_threads();
    omp_set_num_threads(1);
    session.Run(Ort::RunOptions{nullptr}, input_names, &input_val, 1, output_names, &output_val, 1);
    omp_set_num_threads(nt);

    std::cout << "localizer: " << t_.elapsed_ms() << " ms\n";

    return {
        sigmoid(results[0]),
        {
            results[1],
            results[2],
            results[3]-results[1], // Width
            results[4]-results[2] // Height
        }};
}


neuralnet_tracker::PoseEstimator::PoseEstimator(Ort::MemoryInfo &allocator_info, Ort::Session &&session) :
    session{std::move(session)},
    scaled_frame(input_img_height, input_img_width, CV_8U),
    input_mat(input_img_height, input_img_width, CV_32F)
    //output_keypoints(keypoint_dim, num_keypoints)
{
    {
        const std::int64_t input_shape[4] = { 1, 1, input_img_height, input_img_width };
        input_val = Ort::Value::CreateTensor<float>(allocator_info, input_mat.ptr<float>(0), input_mat.total(), input_shape, 4);
    }

    {
        const std::int64_t output_shape[2] = { 1, 3 };
        output_val[0] = Ort::Value::CreateTensor<float>(
            allocator_info, &output_coord[0], output_coord.rows, output_shape, 2);
    }

    {
        const std::int64_t output_shape[2] = { 1, 4 };
        output_val[1] = Ort::Value::CreateTensor<float>(
            allocator_info, &output_quat[0], output_quat.rows, output_shape, 2);
    }

    // {
    //     const std::int64_t output_shape[3] = { 1, keypoint_dim, num_keypoints };
    //     output_val_keypoints = Ort::Value::CreateTensor<float>(
    //         allocator_info, output_keypoints.ptr<float>(), output_keypoints.total(), output_shape, 3);
    // }

    {
        const std::int64_t output_shape[2] = { 1, 4 };
        output_val[2] = Ort::Value::CreateTensor<float>(
            allocator_info, &output_box[0], output_box.rows, output_shape, 2);
    }
}


std::optional<neuralnet_tracker::PoseEstimator::Face> neuralnet_tracker::PoseEstimator::run(
    const cv::Mat &frame, const cv::Rect &box)
{
    cv::Mat cropped;
    
    const int patch_size = std::max(box.width, box.height)*1.05;
    const cv::Point2f patch_center = {
        std::clamp<float>(box.x + 0.5f*box.width, 0.f, frame.cols),
        std::clamp<float>(box.y + 0.5f*box.height, 0.f, frame.rows)
    };
    cv::getRectSubPix(frame, {patch_size, patch_size}, patch_center, cropped);

    if (cropped.rows != patch_size || cropped.cols != patch_size)
        return {};
    
    auto p = input_mat.ptr(0);

    cv::resize(cropped, scaled_frame, { input_img_width, input_img_height }, 0, 0, cv::INTER_AREA);
    //cv::imwrite("/tmp/pose_estimator_input.png", scaled_frame);
    scaled_frame.convertTo(input_mat, CV_32F, 1./255., -0.5);

    assert (input_mat.ptr(0) == p);
    assert (!input_mat.empty() && input_mat.isContinuous());
    assert (input_mat.cols == input_img_width && input_mat.rows == input_img_height);

    const char* input_names[] = {"x"};
    const char* output_names[] = {"pos_size", "quat", "box"};

    Timer t_; t_.start();

    const auto nt = omp_get_num_threads();
    omp_set_num_threads(1);
    session.Run(Ort::RunOptions{nullptr}, input_names, &input_val, 1, output_names, output_val, 3);
    omp_set_num_threads(nt);

    std::cout << "pose net: " << t_.elapsed_ms() << " ms\n";

    // Perform coordinate transformation.
    // From patch-local normalized in [-1,1] to
    // frame unnormalized pixel coordinates.

    const cv::Point2f center = patch_center + 
        (0.5f*patch_size)*cv::Point2f{output_coord[0], output_coord[1]};

    const float size = patch_size*0.5f*output_coord[2];

    // Eigen takes quat components in the order w, x, y, z.
    const Eigen::Quaternionf rotation = { 
        output_quat[3], 
        output_quat[0], 
        output_quat[1], 
        output_quat[2] };

    const cv::Rect2f outbox = {
        patch_center.x + (0.5f*patch_size)*output_box[0],
        patch_center.y + (0.5f*patch_size)*output_box[1],
        0.5f*patch_size*(output_box[2]-output_box[0]),
        0.5f*patch_size*(output_box[3]-output_box[1])
    };

    return std::optional<Face>({
        rotation, cv::Mat_<float>{}, outbox, center, size
    });
}


bool neuralnet_tracker::detect()
{
    // Note: BGR colors!
    if (!last_localizer_roi || !last_roi ||
        iou(*last_localizer_roi,*last_roi)<0.25)
    {
        auto [p, rect] = localizer->run(grayscale);

        const cv::Rect2f pix_rect = unnormalize(rect, frame.rows, frame.cols);

        if (p > 0.5)
        {
            last_localizer_roi = pix_rect;
            last_roi = pix_rect;
        }
    }

    if (!last_roi)
        return false;

    auto face = poseestimator->run(grayscale, *last_roi);
    
    if (!face)
    {
        last_roi.reset();
        return false;
    }
    else
    {
        last_roi = face->box;
    }
    
    if (last_roi) 
    {
        const int col = 255;
        cv::rectangle(frame, *last_roi, cv::Scalar(0, 255, 0), /*thickness=*/1);
    }
    if (last_localizer_roi)
    {
        const int col = 255;
        cv::rectangle(frame, *last_localizer_roi, cv::Scalar(col, 0, 255-col), /*thickness=*/1);
    }

    cv::circle(frame, static_cast<cv::Point>(face->center), int(face->size), cv::Scalar(255,255,255), 2);
    cv::circle(frame, static_cast<cv::Point>(face->center), 3, cv::Scalar(255,255,255), -1);

    const Eigen::Quaternionf rot_correction = compute_rotation_correction(
        normalize(face->center, frame.rows, frame.cols),
        intrinsics.focal_length_w);

    const Eigen::Quaternionf rot = rot_correction*face->rotation;

    Eigen::Matrix3f m = rot.toRotationMatrix();
    auto draw_coord_line = [&](const Eigen::Vector3f &v, const cv::Scalar& color)
    {
        static constexpr float len = 100.f;
        cv::Point q = face->center + len*cv::Point2f{-v[2], -v[1]};
        cv::line(frame, static_cast<cv::Point>(face->center), static_cast<cv::Point>(q), color, 2);
    };
    draw_coord_line(m.col(0), {0, 0, 255});
    draw_coord_line(m.col(1), {0, 255, 0});
    draw_coord_line(m.col(2), {255, 0, 0});

    /*
         
       hhhhhh  <- head size (meters)
      \      | -----------------------
       \     |                         \
        \    |                          |
         \   |                          |- tz (meters)
          ____ <- face->size / width    |
           \ |  |                       |
            \|  |- focal length        /
               ------------------------
    */

    // Compute the location the network outputs in 3d space.
    const Eigen::Vector3f face_world_pos = image_to_world(face->center.x, face->center.y, face->size, head_size_meters);
    // But this is in general not the location of the rotation joint in the neck.
    // So we need an extra offset. Which we determine by solving
    // z,y,z-pos = head_joint_loc + R_face * offset
    const Eigen::Vector3f pos = face_world_pos
        - m * Eigen::Vector3f{
            static_cast<float>(s.offset_fwd), 
            static_cast<float>(s.offset_up),
            0.f};

    { 
        // Draw the computed joint position
        auto xy = world_to_image(pos).eval();
        cv::circle(frame, cv::Point(xy[0],xy[1]), 5, cv::Scalar(0,0,255), -1);
    }

    // Definition of coordinate frame
    //   x: Pointing forward, i.e. toward the camera
    //   y: Up
    //   z: Right, from the perspective of the tracked person.
    // When looking straight ahead the rotation is zero.
    const auto mx = m.col(0);
    const auto my = m.col(1);
    const auto mz = -m.col(2);
    const float yaw = std::atan2(mx[2], mx[0]);
    const float pitch = -std::atan2(-mx[1], std::sqrt(mx[2]*mx[2]+mx[0]*mx[0]));
    const float roll = std::atan2(-my[2], mz[2]);
    {
        QMutexLocker lck(&mtx);
        constexpr double rad2deg = 180/M_PI;
        pose[Yaw]   = rad2deg * yaw;
        pose[Pitch] = rad2deg * pitch;
        pose[Roll]  = rad2deg * roll;

        // convert to cm
        pose[TX] = -pos[2] * 100.;
        pose[TY] = pos[1] * 100.;
        pose[TZ] = -pos[0] * 100.;
    }
    return true;
}


neuralnet_tracker::neuralnet_tracker()
{
    opencv_init();
}


neuralnet_tracker::~neuralnet_tracker()
{
    requestInterruption();
    wait();
    // fast start/stop causes breakage
    portable::sleep(1000);
}


module_status neuralnet_tracker::start_tracker(QFrame* videoframe)
{
    videoframe->show();
    videoWidget = std::make_unique<cv_video_widget>(videoframe);
    layout = std::make_unique<QHBoxLayout>();
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(videoWidget.get());
    videoframe->setLayout(layout.get());
    videoWidget->show();
    start();
    return status_ok();
}


bool neuralnet_tracker::load_and_initialize_model()
{
    // QFile::encodeName
    const QString localizer_model_path_enc =
        OPENTRACK_BASE_PATH+"/" OPENTRACK_I18N_PATH "/../models/head-localizer.onnx";
    const QString poseestimator_model_path_enc =
        OPENTRACK_BASE_PATH+"/" OPENTRACK_I18N_PATH "/../models/head-pose.onnx";

    try
    {
        env = Ort::Env{
            OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
            "tracker-neuralnet"
        };
        auto opts = Ort::SessionOptions{};
        opts.SetIntraOpNumThreads(1); // TODO use openmp control modes?
        opts.SetInterOpNumThreads(1);
        opts.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        //opts.DisablePerSessionThreads();
        opts.EnableCpuMemArena();
        allocator_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        localizer.emplace(
            allocator_info, 
            Ort::Session{env, convert(localizer_model_path_enc).c_str(), opts});
        
        poseestimator.emplace(
            allocator_info,
            Ort::Session{env, convert(poseestimator_model_path_enc).c_str(), opts});
    }
    catch (const Ort::Exception &e)
    {
        std::cerr << e.what() << std::endl;
        return false;
    }
    return true;
}


bool neuralnet_tracker::open_camera()
{
    int fps = enum_to_fps(s.force_fps);

    QMutexLocker l(&camera_mtx);

    camera = video::make_camera(s.camera_name);

    if (!camera)
        return false;

    video::impl::camera::info args {};

    args.width = 320;
    args.height = 240;

    if (fps)
        args.fps = fps;

    if (!camera->start(args))
    {
        qDebug() << "neuralnet tracker: can't open camera";
        return false;
    }
    return true;
}


void neuralnet_tracker::set_intrinsics()
{
    const int w = grayscale.cols, h = grayscale.rows;
    const double diag_fov = s.fov * M_PI / 180.;
    const double fov_w = 2.*atan(tan(diag_fov/2.)/sqrt(1. + h/(double)w * h/(double)w));
    const double fov_h = 2.*atan(tan(diag_fov/2.)/sqrt(1. + w/(double)h * w/(double)h));
    const double focal_length_w = 1. / tan(.5 * fov_w);
    const double focal_length_h = 1. / tan(.5 * fov_h);

    intrinsics.fov_h = fov_h;
    intrinsics.fov_w = fov_w;
    intrinsics.focal_length_w = focal_length_w;
    intrinsics.focal_length_h = focal_length_h;
}


Eigen::Vector3f neuralnet_tracker::image_to_world(float x, float y, float size, float real_size) const
{
    // Compute the location the network outputs in 3d space.
    const float xpos = -(intrinsics.focal_length_w * frame.cols * 0.5f) / size * real_size;
    const float zpos = (x / frame.cols * 2.f - 1.f) * xpos / intrinsics.focal_length_w;
    const float ypos = (y / frame.rows * 2.f - 1.f) * xpos / intrinsics.focal_length_h;
    return {xpos, ypos, zpos};
}


Eigen::Vector2f neuralnet_tracker::world_to_image(const Eigen::Vector3f& pos) const
{
    const float xscr = pos[2] / pos[0] * intrinsics.focal_length_w;
    const float yscr = pos[1] / pos[0] * intrinsics.focal_length_h;
    const float x = (xscr+1.)*0.5f*frame.cols;
    const float y = (yscr+1.)*0.5f*frame.rows;
    return {x, y};
}


void neuralnet_tracker::update_fps()
{
    const double dt = fps_timer.elapsed_seconds();
    fps_timer.start();
    const double alpha = dt/(dt + RC);

    if (dt > 1e-3)
    {
        fps *= 1 - alpha;
        fps += alpha * (1./dt + .8);
    }
}


void neuralnet_tracker::run()
{
    if (!open_camera())
        return;

    if (!load_and_initialize_model())
        return;

    fps_timer.start();

    while (!isInterruptionRequested())
    {
        {
            QMutexLocker l(&camera_mtx);

            auto [ img, res ] = camera->get_frame();

            if (!res)
            {
                portable::sleep(100);
                continue;
            }

            auto color = cv::Mat(img.height, img.width, CV_8UC(img.channels), (void*)img.data, img.stride);
            color.copyTo(frame);

            switch (img.channels)
            {
            case 1:
                grayscale.setTo(color); 
                break;
            case 3:
                cv::cvtColor(color, grayscale, cv::COLOR_BGR2GRAY);
                break;
            default:
                qDebug() << "Can't handle" << img.channels << "color channels";
                return;
            }
        }

        set_intrinsics();

        update_fps();

        detect();

        std::cout << "fps = " << fps << "\n";

        if (frame.rows > 0)
            videoWidget->update_image(frame);
    }
}


void neuralnet_tracker::data(double *data)
{
    QMutexLocker lck(&mtx);

    data[Yaw] = pose[Yaw];
    data[Pitch] = pose[Pitch];
    data[Roll] = pose[Roll];
    data[TX] = pose[TX];
    data[TY] = pose[TY];
    data[TZ] = pose[TZ];
}


void neuralnet_dialog::make_fps_combobox()
{
    std::vector<std::tuple<int, int>> resolutions;
    resolutions.reserve(fps_MAX);

    for (int k = 0; k < fps_MAX; k++)
    {
        int hz = enum_to_fps(k);
        resolutions.emplace_back(k, hz);
    }

    std::sort(resolutions.begin(), resolutions.end(), [](const auto& a, const auto& b) {
        auto [idx1, hz1] = a;
        auto [idx2, hz2] = b;

        return hz1 < hz2;
    });

    for (auto [idx, hz] : resolutions)
    {
        QString name;

        if (hz == 0)
            name = tr("Default");
        else
            name = QString::number(hz);

        ui.cameraFPS->addItem(name, idx);
    }
}


neuralnet_dialog::neuralnet_dialog()
{
    ui.setupUi(this);

    make_fps_combobox();
    tie_setting(s.force_fps, ui.cameraFPS);

    for (const auto& str : video::camera_names())
        ui.cameraName->addItem(str);

    tie_setting(s.camera_name, ui.cameraName);
    tie_setting(s.fov, ui.cameraFOV);
    tie_setting(s.offset_fwd, ui.cx);
    tie_setting(s.offset_up, ui.cy);

    connect(ui.buttonBox, SIGNAL(accepted()), this, SLOT(doOK()));
    connect(ui.buttonBox, SIGNAL(rejected()), this, SLOT(doCancel()));
    connect(ui.camera_settings, SIGNAL(clicked()), this, SLOT(camera_settings()));

    connect(&s.camera_name, value_::value_changed<QString>(), this, &neuralnet_dialog::update_camera_settings_state);

    update_camera_settings_state(s.camera_name);
}


void neuralnet_dialog::doOK()
{
    s.b->save();
    close();
}


void neuralnet_dialog::doCancel()
{
    close();
}


void neuralnet_dialog::camera_settings()
{
    if (tracker)
    {
        QMutexLocker l(&tracker->camera_mtx);
        (void)tracker->camera->show_dialog();
    }
    else
        (void)video::show_dialog(s.camera_name);
}


void neuralnet_dialog::update_camera_settings_state(const QString& name)
{
    (void)name;
    ui.camera_settings->setEnabled(true);
}


settings::settings() : opts("neuralnet-tracker") {}


OPENTRACK_DECLARE_TRACKER(neuralnet_tracker, neuralnet_dialog, neuralnet_metadata)
