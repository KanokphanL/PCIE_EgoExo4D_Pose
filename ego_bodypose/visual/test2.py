import numpy as np
from vispy import scene
from vispy.scene import visuals
from vispy.visuals.transforms import STTransform
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget
import sys

class MyCanvas:
    def __init__(self, layout):  # 使用传入的 layout
        ############### 创建 Canvas ###############
        self.canvas = scene.SceneCanvas(keys="interactive", bgcolor="black", size=(800, 600))
        self.canvas.show()
        
        # 添加 Canvas 到布局
        layout.addWidget(self.canvas.native)

        ############### 创建 2D Layer ###############
        self.view_2D = scene.Widget(parent=self.canvas.scene, name='2D_View')
        self.view_2D.camera = scene.PanZoomCamera(aspect=1)
        self.view_2D.camera.set_range(x=(0, 200), y=(0, 200))
        self.view_2D.order = 1  # 图层优先级：0 表示最底层

        # 在 2D View 上绘制一个简单的散点图
        self.scatter = visuals.Markers()
        self.scatter.set_data(pos=np.array([[200, 300], [400, 500], [600, 100]]),
                              face_color='red', size=10)
        self.view_2D.add(self.scatter)

        ############### 创建 3D Layer ###############
        self.view_3D = scene.Widget(parent=self.canvas.scene, name='3D_View')
        self.view_3D.camera = "turntable"
        self.view_3D.order = 0  # 图层优先级：1 表示在 2D View 上方

        # 在 3D View 上绘制一个简单的立方体
        # self.cube = visuals.Cube(edge_color='blue')
        # self.cube.transform = STTransform(translate=(0, 0, -5))
        # self.view_3D.add(self.cube)
        line = scene.visuals.Line(np.random.rand(100, 3), parent=self.view_3D.scene)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QWidget()
    layout = QVBoxLayout(window)
    
    canvas_app = MyCanvas(layout)
    window.show()
    sys.exit(app.exec_())
