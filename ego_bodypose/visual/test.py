from vispy import scene
from vispy.geometry import Rect
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget
import sys
import numpy as np

class MyCanvas:
    def __init__(self, layout):
        ############### 创建 Canvas ###############
        self.canvas = scene.SceneCanvas(keys="interactive", bgcolor="black", size=(800, 600))
        self.canvas.show()

        # 将 Canvas 添加到传入的布局中
        layout.addWidget(self.canvas.native)
        
        # # 创建一个根 Widget
        # self.grid = scene.widgets.Grid(parent=self.canvas.scene)
        # self.grid.rect = Rect(0, 0, 800, 600)  # 设置大小为 800x600
        # 创建一个 Grid 并设置为 central_widget 的子组件
        self.grid = scene.widgets.Grid(parent=self.canvas.central_widget)
        
        ############### 3D 视图 ###############
        self.view_3D = scene.widgets.ViewBox(border_color="white", parent=self.grid)
        self.view_3D.camera = "turntable"
        # self.grid.add_widget(self.view_3D, row=0, col=1)  # 将 3D 视图放在主区域
        self.grid.add_widget(self.view_3D, row=0, col=1, row_span=1, col_span=3)  # 占据大部分区域


        # ############### 2D 视图 ###############
        self.view_2D = scene.widgets.ViewBox(border_color="white", parent=self.grid)
        self.view_2D.camera = scene.PanZoomCamera(aspect=1)
        self.view_2D.camera.set_range(x=(200, 400), y=(200, 400))
        self.grid.add_widget(self.view_2D, row=0, col=0) # 将 2D 视图放在左上角
        
        img_data = np.random.rand(512, 512, 3) * 255  # 生成随机的 RGB 图像数据
        img_data = img_data.astype(np.uint8)  # 转换为整数类型，以符合图像格式
        

        image = scene.visuals.Image(img_data, parent=self.view_2D.scene)
        line = scene.visuals.Line(np.random.rand(100, 3), parent=self.view_3D.scene)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QWidget()
    layout = QVBoxLayout(window)

    canvas_app = MyCanvas(layout)
    layout.addWidget(canvas_app.canvas.native)

    window.show()
    sys.exit(app.exec_())
