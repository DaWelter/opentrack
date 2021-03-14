#pragma once
#include <QtGlobal>
#include <QObject>
#include <QStringList>
#include <QIODevice>
#include <QTextCodec>
#include <QRegExp>
#include <QtGlobal>

class CSV
{
public:
    QString readLine();
    bool parseLine(QStringList& ret);

    static bool getGameData(int gameID, unsigned char* table, QString& gamename);
private:
    CSV(QIODevice* device);
    static QTextCodec& getCodec();

    QIODevice* m_device;
    QString m_string;
    int m_pos;

    static void initCodec();
    static const QRegExp m_rx, m_rx2;
};
