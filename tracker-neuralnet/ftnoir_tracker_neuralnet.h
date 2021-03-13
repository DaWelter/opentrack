/* Copyright (c) 2013-2015 Stanislaw Halik <sthalik@misaki.pl>
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 */

#pragma once

#include "options/options.hpp"
#include "api/plugin-api.hpp"
#include "cv/video-widget.hpp"
#include "compat/timer.hpp"
#include "video/camera.hpp"

#include <QObject>
#include <QThread>
#include <QMutex>
#include <QHBoxLayout>
#include <QDialog>
#include <QTimer>

#include <memory>
#include <cinttypes>

#include <onnxruntime_cxx_api.h>

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "ui_neuralnet-trackercontrols.h"

using namespace options;

enum aruco_fps
{
    fps_default = 0,
    fps_30      = 1,
    fps_60      = 2,
    fps_75      = 3,
    fps_125     = 4,
    fps_200     = 5,
    fps_50      = 6,
    fps_100     = 7,
    fps_120     = 8,
    fps_300     = 9,
    fps_250     = 10,
    fps_MAX     = 11,
};

struct settings : opts {
    value<double> offset_fwd { b, "offset-fwd", 0.2 },
                  offset_up { b, "offset-up", 0 };

    value<QString> camera_name { b, "camera-name", ""};
    value<int> fov { b, "field-of-view", 56 };
    value<aruco_fps> force_fps { b, "force-fps", fps_default };

    settings();
};

class neuralnet_tracker : protected virtual QThread, public ITracker
{
    Q_OBJECT
public:
    neuralnet_tracker();
    ~neuralnet_tracker() override;
    module_status start_tracker(QFrame* frame) override;
    void data(double *data) override;
    void run() override;

    // void getRT(cv::Matx33d &r, cv::Vec3d &t);
    QMutex camera_mtx;
    std::unique_ptr<video::impl::camera> camera;

private:
    bool detect();
    bool open_camera();
    void set_intrinsics();
    void update_fps();
    bool load_and_initialize_model();
    Eigen::Vector3f image_to_world(float x, float y, float size, float real_size) const;
    Eigen::Vector2f world_to_image(const Eigen::Vector3f& p) const;

    struct CamIntrinsics
    {
        float focal_length_w;
        float focal_length_h;
        float fov_w;
        float fov_h;
    };

    class Localizer
    {
        public:
            Localizer(Ort::MemoryInfo &allocator_info,
                      Ort::Session &&session);
            
            // Returns bounding box normalized to [-1,1].
            std::pair<float, cv::Rect2f> run(
                const cv::Mat &frame);

        private:
            inline static constexpr int input_img_width = 288;
            inline static constexpr int input_img_height = 224;
            Ort::Session session{nullptr};
            cv::Mat scaled_frame{}, input_mat{};
            Ort::Value input_val{nullptr}, output_val{nullptr};
            std::array<float, 5> results;
    };

    class PoseEstimator
    {
        public:
            struct Face
            {
                Eigen::Quaternionf rotation;
                cv::Mat_<float> keypoints;
                cv::Rect2f box;
                cv::Point2f center;
                float size;
            };

            PoseEstimator(Ort::MemoryInfo &allocator_info,
                          Ort::Session &&session);
            std::optional<Face> run(const cv::Mat &frame, const cv::Rect &box);

        private:
            inline static constexpr int input_img_width = 129;
            inline static constexpr int input_img_height = 129;
            inline static constexpr int num_keypoints = 68;
            inline static constexpr int keypoint_dim = 3;
            Ort::Session session{nullptr};
            cv::Mat scaled_frame{}, input_mat{};
            Ort::Value input_val{nullptr};
            cv::Vec<float, 3> output_coord{};
            cv::Vec<float, 4> output_quat{};
            //cv::Mat_<float> output_keypoints{};
            cv::Vec<float, 4> output_box{};
            Ort::Value output_val[3] = {
                Ort::Value{nullptr}, 
                Ort::Value{nullptr}, 
                Ort::Value{nullptr}};
    };

    QMutex mtx;
    std::unique_ptr<cv_video_widget> videoWidget;
    std::unique_ptr<QHBoxLayout> layout;
    settings s;
    double pose[6] {}, fps = 0;
    CamIntrinsics intrinsics{};
    cv::Mat frame, grayscale;
    std::optional<cv::Rect2f> last_localizer_roi;
    std::optional<cv::Rect2f> last_roi;
    Timer fps_timer;
    float head_size_meters = 0.2;

    Ort::Env env{nullptr};
    Ort::MemoryInfo allocator_info{nullptr};

    std::optional<Localizer> localizer;
    std::optional<PoseEstimator> poseestimator;

    static constexpr double RC = .25;
};

class neuralnet_dialog : public ITrackerDialog
{
    Q_OBJECT
public:
    neuralnet_dialog();
    void register_tracker(ITracker * x) override { tracker = static_cast<neuralnet_tracker*>(x); }
    void unregister_tracker() override { tracker = nullptr; }
private:
    void make_fps_combobox();

    Ui::Form ui;
    neuralnet_tracker* tracker = nullptr;
    settings s;
private Q_SLOTS:
    void doOK();
    void doCancel();
    void camera_settings();
    void update_camera_settings_state(const QString& name);
};

class neuralnet_metadata : public Metadata
{
    Q_OBJECT
    QString name() override { return QString("neuralnet tracker"); }
    QIcon icon() override { return QIcon(":/images/neuralnet.png"); }
};
