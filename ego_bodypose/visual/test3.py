from vispy import scene
from vispy.scene import visuals
from vispy.visuals.transforms import STTransform
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget
import numpy as np
import sys
from vispy.geometry import Rect

class MyCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D & 3D Layer Overlay Example")
        self.resize(800, 600)

        # 创建一个布局
        self.layout = QVBoxLayout(self)
        
        # 创建一个 VisPy Canvas
        self.canvas = scene.SceneCanvas(keys="interactive", bgcolor="black", size=(800, 800))
        self.canvas.show()

        # 把 Canvas 放到 PyQt 的布局中
        self.layout.addWidget(self.canvas.native)

        # 创建一个 Grid 来管理多个 ViewBox
        self.grid = self.canvas.central_widget.add_grid()

        ############### 创建 3D Layer ###############
        self.view_3D = self.grid.add_view(row=0, col=0)
        # self.view_3D.camera = scene.cameras.TurntableCamera(fov=60, distance=10)
        self.view_3D.camera = "turntable"
        
        # 在 3D View 上绘制一个立方体
        self.cube = visuals.Cube(edge_color='blue', parent=self.view_3D.scene)
        self.cube.transform = STTransform(translate=(0, 0, 0))

        ############### 创建 2D Layer ###############
        # self.view_2D = self.grid.add_view(row=0, col=0)
        # self.view_2D.camera = scene.PanZoomCamera(aspect=1)
        # self.view_2D.border_color = (1, 0, 0, 1)  # 用红色边框标出 2D 区域（便于调试）
        # self.view_2D.camera.set_range(x=(-200, 200), y=(-200, 200))  # 固定视图范围
        # self.view_2D.rect = Rect(-1000, -1000, 2000, 2000)  # 固定显示区域
        
        
        # 包装一个 Widget 用于控制大小
        tmp = self.grid.add_view(row=0, col=0)
        self.widget_2D = scene.Widget(parent=tmp.scene)
        self.widget_2D.pos = (0, 0)  # 放置在左上角
        self.widget_2D.size = (100, 100)  # 固定大小为 200x200
        
        img_data = np.ones((300, 200, 3), dtype=np.uint8) * 255
        img_data[10:99, 10:49, :] = 0
        self.image = scene.visuals.Image(img_data, parent=self.widget_2D, method='auto')
        # 设置 Image 的位置与缩放
        self.image.transform = STTransform(translate=(0, 0), scale=(self.widget_2D.size[0]/img_data.shape[1], self.widget_2D.size[1]/img_data.shape[0]))

        # # 创建 view_2D 放在 Widget 内
        # self.view_2D = scene.ViewBox(parent=self.widget_2D)
        # self.view_2D.camera = scene.PanZoomCamera(aspect=1)
        # self.view_2D.border_color = (1, 0, 0, 1)  # 用红色边框标出 2D 区域（便于调试）
        # self.view_2D.camera.set_range(x=(-1000, 1000), y=(-1000, 1000))  # 固定显示范围

        # # 在 2D View 上绘制一个散点图
        # # img_data = np.random.rand(512, 512, 3) * 255  # 生成随机的 RGB 图像数据
        # # img_data = img_data.astype(np.uint8)  # 转换为整数类型，以符合图像格式
        
        # img_data = np.ones((100, 50, 3), dtype=np.uint8) * 255
        # img_data[10:99, 10:49, :] = 0
        # # img_data = np.tile(np.arange(100, 256, 10).reshape(1, -1, 1), (10, 1, 3))
        # self.image = scene.visuals.Image(img_data, parent=self.view_2D.scene)
        # # 缩放图片使其适配 200x200 区域
        # self.image.transform = STTransform(scale=(200 / img_data.shape[1], 200 / img_data.shape[0]), translate=(0, 0))
        
        # self.scatter = visuals.Markers(parent=self.view_2D.scene)
        # self.scatter.set_data(pos=np.array([[0, 0], [100, 100], [-100, -100]]),
        #                       face_color='red', size=10)
        

        # 确保 2D 图层在 3D 图层上方显示
        # self.view_2D.order = 0
        self.view_3D.order = 1

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyCanvas()
    window.show()
    sys.exit(app.exec_())
