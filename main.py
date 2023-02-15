import ctypes
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from tkinter import Tk, PanedWindow, LabelFrame, Label, PhotoImage, Button, Scrollbar, Text, ttk, Toplevel
from tkinter import filedialog
from tkinter import messagebox

import cv2
from PIL import ImageTk, Image

from detect_and_recognition import detect_and_text


def base_path(path):
    if getattr(sys, 'frozen', None):
        basedir = sys._MEIPASS
    else:
        basedir = os.path.dirname(__file__)
    return os.path.join(basedir, path).replace("\\", "/")


def check_event(cur_future, call, wid):
    if cur_future is None:
        return
    else:
        if cur_future.done():
            if wid:
                wid.destroy()
            call(*cur_future.result())
            bt4.pack()
        else:
            root.after(100, check_event, cur_future, call, wid)


def file_open():
    global log, glb_vid_new, glb_vid_ori, future
    bmp_path = filedialog.askopenfilename(
        filetypes=[("所有文件", "*.*"), ("BMP", ".bmp"), ("JPEG", ".jpg"), ("PNG", ".png")])
    if bmp_path != "":
        log.config(state='normal')
        log.insert('end', "读取%s\n" % bmp_path)
        log.config(state='disabled')
        if bmp_path.endswith(".mp4"):
            messagebox.showinfo('提示', "请指定输出文件夹")
            output_dir = filedialog.askdirectory()
            if output_dir != "":
                new_window = Toplevel(root)
                new_window.title("加载进度条")
                screen_width = root.winfo_screenwidth() / 2
                screen_height = root.winfo_screenheight() / 2
                new_window.geometry(f"+{int(screen_width)}+{int(screen_height)}")
                progressbarOne = ttk.Progressbar(new_window)
                bar_text = Label(new_window, text="加载进度", font=("微软雅黑", 20, 'bold'))
                bar_text.pack(fill='x')
                # 进度值最大值
                progressbarOne['maximum'] = 100
                # 进度值初始值
                progressbarOne['value'] = 0
                progressbarOne.pack(fill="both", padx=70, pady=50)
                future = thread_pool.submit(detect_and_text, path=bmp_path, output_dir=output_dir, log=log,
                                            progressbar=progressbarOne, bar_text=bar_text)
                check_event(future, video_play, new_window)
        else:
            success, plate_texts = show(bmp_path)
            log.config(state='normal')
            log.insert('end', "%s\n" % "读取成功" if success else "读取失败")
            log.insert('end', "识别结果：%s\n" % plate_texts)
            log.config(state='disabled')
            messagebox.showinfo("识别结果：", "\n".join(plate_texts))


def dir_open():
    dirname = filedialog.askdirectory()
    file_list = []
    for file in os.listdir(dirname):
        if os.path.splitext(file)[1] in [".jpg", ".png", ".bmp"]:
            file_list.append(os.path.join(dirname, file))
    messagebox.showinfo('提示', "请指定输出文件夹")
    output_dir = filedialog.askdirectory()
    log.config(state='normal')
    log.insert('end', '正在飞速处理,请稍等!!!!\n')
    log.config(state='disabled')
    log.update()
    if dirname != "":
        plate_text = detect_and_text(file_list, output_dir=output_dir)
        log.config(state='normal')
        log.insert('end', '识别结果：%s' % plate_text)
        log.config(state='disabled')
        messagebox.showinfo("识别结果", '输出图片已保存至%s\n' % output_dir)
        os.startfile(output_dir)


def file_save():
    if yuv_rgb_img is None:
        messagebox.showwarning("警告", "尚未生成识别结果，无法保存")
        return
    else:
        yuv_path = filedialog.asksaveasfilename(filetypes=[("JPG", ".jpg"), ("PNG", ".png"), ("BMP", ".bmp"), ],
                                                defaultextension='.bmp')
        if yuv_path != "":
            log.config(state='normal')
            log.insert("end", "保存文件至%s\n" % yuv_path)
            log.config(state='disabled')
            yuv_path = yuv_path.replace("/", "\\\\")
            yuv_rgb_img.convert('RGB').save(yuv_path)
            messagebox.showinfo("信息", "保存成功")


def show(bmp_path):
    global bmp_img, bmp_photo, yuv_img, yuv_photo, yuv_rgb_img, bt2
    img_ori, yuv_rgb_img, plate_texts = detect_and_text(bmp_path)
    width = bmp_frame.winfo_width()
    w, h = img_ori.size
    bmp_photo = ImageTk.PhotoImage(img_ori.resize((width, round(width / w * h))))
    bmp_img.config(image=bmp_photo)
    bmp_img.place(relx=0.5, rely=0.5, anchor='center')

    yuv_photo = ImageTk.PhotoImage(yuv_rgb_img.resize((width, round(width / w * h))))
    yuv_img.config(image=yuv_photo)
    yuv_img.place(relx=0.5, rely=0.5, anchor='center')
    bt2.pack()
    return True, plate_texts


def video_play(vid_ori, vid_new):
    global bmp_img, yuv_img, th, bmp_frame
    vid_ori = cv2.VideoCapture(vid_ori)
    vid_new = cv2.VideoCapture(vid_new)
    width = bmp_frame.winfo_width()
    while vid_ori.isOpened():
        ret, frame = vid_ori.read()  # 读取照片
        ret_new, frame_new = vid_new.read()  # 读取照片
        if ret and ret_new:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换颜色使播放时保持原有色彩
            img = Image.fromarray(img)
            w, h = img.size
            current_image = img.resize((width, round(width / w * h)))  # 将图像转换成Image对象
            imgtk = ImageTk.PhotoImage(image=current_image)
            bmp_img.config(image=imgtk)
            img = cv2.cvtColor(frame_new, cv2.COLOR_BGR2RGB)  # 转换颜色使播放时保持原有色彩
            current_image = Image.fromarray(img).resize((width, round(width / w * h)))  # 将图像转换成Image对象
            imgtk_new = ImageTk.PhotoImage(image=current_image)
            yuv_img.config(image=imgtk_new)
            time.sleep(1 / vid_ori.get(cv2.CAP_PROP_FPS))
            bmp_img.update()  # 每执行以此只显示一张图片，需要更新窗口实现视频播放
            yuv_img.update()
        else:
            vid_ori.release()
            vid_new.release()
            break


glb_vid_ori, glb_vid_new = None, None
future = None
thread_pool = ThreadPoolExecutor()
root = Tk()
# 告诉操作系统使用程序自身的dpi适配
ctypes.windll.shcore.SetProcessDpiAwareness(1)
# 获取屏幕的缩放因子
ScaleFactor = ctypes.windll.shcore.GetScaleFactorForDevice(0)
# 设置程序缩放
root.tk.call('tk', 'scaling', (ScaleFactor // 75))
root.title("基于YOLOv5的车牌号检测以及识别")
root.geometry("%dx%d+100+50" % (root.winfo_screenwidth(), root.winfo_screenheight()))
root.iconbitmap(base_path("assets/icon.ico"))
root.update()

m1 = PanedWindow(root, orient='vertical', width=root.winfo_width() / 3, showhandle=True)
m1.pack(fill='both', side='left')
menu_frame = LabelFrame(m1, bg='white')
menu_frame.grid(row=0)

text = Label(menu_frame, text="这是菜单栏", font=("微软雅黑", 20, 'bold'))
text.pack(fill='x')
open_photo = PhotoImage(file=base_path("assets/folder.png"))
bt1 = Button(menu_frame, text="选择文件", activebackground="red", bd=4, font=("微软雅黑"), bg='yellow',
             command=file_open, image=open_photo,
             compound='left', padx=10)
bt1.pack()

bt3 = Button(menu_frame, text="选择文件夹", activebackground="red", bd=4, font=("微软雅黑"), bg='yellow',
             command=dir_open, image=open_photo,
             compound='left', padx=10)
bt3.pack()

save_photo = PhotoImage(file=base_path("assets/save.png"))
bt2 = Button(menu_frame, text="保存识别结果", bd=4, font=("微软雅黑"), bg='yellow', command=file_save, image=save_photo,
             compound='left', padx=10)
bt2.pack_forget()

bt4 = Button(menu_frame, text="重新播放", activebackground="red", bd=4, font=("微软雅黑"), bg='yellow',
             command=lambda: check_event(future, video_play, None), image=open_photo,
             compound='left', padx=10)
bt4.pack_forget()

m1.add(menu_frame, height=root.winfo_height() // 2)
info_frame = LabelFrame(m1, bg='white')
info_frame.grid(row=1)
text = Label(info_frame, text="这是日志栏", font=("微软雅黑", 20, 'bold'))
text.pack(fill='x')
scroll = Scrollbar(info_frame, width=20)

scroll.pack(side='right', fill='y')
log = Text(info_frame, font=('宋体', 20, 'normal'))
log.pack(fill='both', expand=1)

log.config(yscrollcommand=scroll.set, state='disabled')
scroll.config(command=log.yview)

m1.add(info_frame)

m2 = PanedWindow(root, orient='vertical', showhandle=True, sashrelief="sunken")
m2.pack(fill='both', side='right', expand=1)

bmp_frame = LabelFrame(m2, bg='white')
# imgs = utils.get_bmp("C:/Users/HUAWEI/Desktop/多媒体作业/couple.bmp")
bmp_img = Label(bmp_frame, text='原图', font=('微软雅黑', 20, 'bold'), compound="bottom")
bmp_img.pack()
m2.add(bmp_frame, height=root.winfo_height() / 2)
# imgs = utils.get_bmp("hallforeground.bmp")
yuv_frame = LabelFrame(m2, bg='white')
yuv_img = Label(yuv_frame, text='识别结果图', font=('微软雅黑', 20, 'bold'), compound="bottom")
# yuv_img.config(image=imgs)
yuv_img.pack(fill='y')
m2.add(yuv_frame)


def main():
    root.mainloop()


if __name__ == "__main__":
    main()
