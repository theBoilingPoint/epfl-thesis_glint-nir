#include "mainwindow.h"
#include <ui_mainwindow.h>
#include "cameracontrolshelp.h"


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->mygl->setFocus();

    connect(ui->redSlider, SIGNAL(valueChanged(int)), ui->mygl, SLOT(slot_setRed(int)));
    connect(ui->greenSlider, SIGNAL(valueChanged(int)), ui->mygl, SLOT(slot_setGreen(int)));
    connect(ui->blueSlider, SIGNAL(valueChanged(int)), ui->mygl, SLOT(slot_setBlue(int)));

    connect(ui->metallicSlider, SIGNAL(valueChanged(int)), ui->mygl, SLOT(slot_setMetallic(int)));
    connect(ui->roughnessSlider, SIGNAL(valueChanged(int)), ui->mygl, SLOT(slot_setRoughness(int)));
    connect(ui->aoSlider, SIGNAL(valueChanged(int)), ui->mygl, SLOT(slot_setAO(int)));
    connect(ui->displacementSpinBox, SIGNAL(valueChanged(double)), ui->mygl, SLOT(slot_setDisplacement(double)));

    connect(ui->envMapButton, SIGNAL(clicked()), ui->mygl, SLOT(slot_loadEnvMap()));
    connect(ui->sceneButton, SIGNAL(clicked()), ui->mygl, SLOT(slot_loadScene()));
    connect(ui->objButton, SIGNAL(clicked()), ui->mygl, SLOT(slot_loadOBJ()));
    connect(ui->sphereButton, SIGNAL(clicked()), ui->mygl, SLOT(slot_revertToSphere()));
    connect(ui->saveImageButton, SIGNAL(clicked()), ui->mygl, SLOT(slot_saveImage()));

    connect(ui->glintCheckBox, SIGNAL(clicked(bool)), ui->mygl, SLOT(slot_setUseGlint(bool)));
    connect(ui->screenSpaceScaleSpinBox, SIGNAL(valueChanged(double)), ui->mygl, SLOT(slot_setScreenSpaceScale(double)));
    connect(ui->logMicrofacetDensitySpinBox, SIGNAL(valueChanged(double)), ui->mygl, SLOT(slot_setLogMicrofacetDensity(double)));
    connect(ui->densityRandomizationSpinBox, SIGNAL(valueChanged(double)), ui->mygl, SLOT(slot_setDensityRandomization(double)));

    connect(ui->rotationXSpinBox, SIGNAL(valueChanged(double)), ui->mygl, SLOT(slot_setRotationX(double)));
    connect(ui->rotationYSpinBox, SIGNAL(valueChanged(double)), ui->mygl, SLOT(slot_setRotationY(double)));
    connect(ui->rotationZSpinBox, SIGNAL(valueChanged(double)), ui->mygl, SLOT(slot_setRotationZ(double)));

    connect(ui->backgroundColourComboBox, SIGNAL(currentIndexChanged(int)), ui->mygl, SLOT(slot_changeBackgroundColour(int)));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_actionQuit_triggered()
{
    QApplication::exit();
}

void MainWindow::on_actionCamera_Controls_triggered()
{
    CameraControlsHelp* c = new CameraControlsHelp();
    c->show();
}
