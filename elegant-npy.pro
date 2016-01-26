TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    writer.cpp \
    reader.cpp \
    common.cpp

HEADERS += \
    writer.h \
    reader.h \
    common.h \
    typehelper.h \
    typehelpers.h \
    npy.h
