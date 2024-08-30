import matplotlib.pyplot as plt
from matplotlib import animation


class Animation:
    JS = 0
    HTML = 1
    ANIMATION_MODE = HTML

    def __init__(self, frames, fps=30):
        """_summary_

        Args:
            frames (np.ndarray): _description_
        """
        self.frames = frames
        self.fps = fps
        self.anim_obj = None
        self.anim_str = None

    def render(self):
        size = (self.frames.shape[2], self.frames.shape[1])
        self.fig = plt.figure(figsize=size, dpi=1)
        plt.axis('off')
        img = plt.imshow(self.frames[0], cmap='gray', vmin=0, vmax=255)
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.anim_obj = animation.FuncAnimation(
            self.fig,
            lambda i: img.set_data(self.frames[i, :, :, :]),
            frames=self.frames.shape[0],
            interval=1000 / self.fps
        )
        plt.close()
        if Animation.ANIMATION_MODE == Animation.HTML:
            self.anim_str = self.anim_obj.to_html5_video()
        elif Animation.ANIMATION_MODE == Animation.JS:
            self.anim_str = self.anim_obj.to_jshtml()
        return self.anim_obj

    def _repr_html_(self):
        if self.anim_obj is None:
            self.render()
        return self.anim_str
