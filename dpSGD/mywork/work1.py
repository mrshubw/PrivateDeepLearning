import dpSGD.dpSGD_MNIST.DPSGD_CNN as dp
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import random

# 创建前端
root = tkinter.Tk()
root.title('作业一')
# root.geometry('1000x400')
root.resizable(width=True, height=True)

# 显示当前工作的信息
work_information = tkinter.StringVar()
work_information.set('固定epsilon和delta，然后画出测试集准确率\n和privacy loss随epoch变化的曲线图')
label_work = tkinter.Label(root, textvariable=work_information,
                           font=('华文行楷', 20), fg='blue', bg='pink')
label_work.grid(row=0, column=0, columnspan=2, padx=5, pady=5)


# sigma按钮的命令函数
def sigma_constant():
    entry_epsilon['state'] = 'normal'
    entry_delta['state'] = 'normal'


def sigma_square():
    entry_epsilon['state'] = 'disabled'
    entry_delta['state'] = 'disabled'


# 输入sigma
sigma_type = tkinter.IntVar()
sigma_type.set(0)
radio_sigma0 = tkinter.Radiobutton(root, text='sigma=constant', variable=sigma_type,
                                   value=0, command=sigma_constant)
radio_sigma0.grid(row=1, column=0, sticky='e')
radio_sigma1 = tkinter.Radiobutton(root, text='sigma=square(gradient)', variable=sigma_type,
                                   value=1, command=sigma_square)
radio_sigma1.grid(row=1, column=1, sticky='w')

# 输入epsilon
label_epsilon = tkinter.Label(root, text='epsilon:')
label_epsilon.grid(row=2, column=0, sticky='e')
text_epsilon = tkinter.StringVar()
text_epsilon.set('1')
entry_epsilon = tkinter.Entry(root, textvariable=text_epsilon, bd=3)
entry_epsilon.grid(row=2, column=1, sticky='w')

# 输入delta
label_delta = tkinter.Label(root, text='delta:')
label_delta.grid(row=3, column=0, sticky='e')
text_delta = tkinter.StringVar()
text_delta.set('0.00001')
entry_delta = tkinter.Entry(root, textvariable=text_delta, bd=3)
entry_delta.grid(row=3, column=1, sticky='w')

# 输入clip_bound
label_clip = tkinter.Label(root, text='clip_bound:')
label_clip.grid(row=4, column=0, sticky='e')
text_clip = tkinter.StringVar()
text_clip.set('0.01')
entry_clip = tkinter.Entry(root, textvariable=text_clip, bd=3)
entry_clip.grid(row=4, column=1, sticky='w')

# 输入batch_size
label_batch = tkinter.Label(root, text='batch size:')
label_batch.grid(row=5, column=0, sticky='e')
text_batch = tkinter.IntVar()
spin_batch = tkinter.Spinbox(root, values=(100, 200, 300, 600, 1000), textvariable=text_batch)
text_batch.set(600)
spin_batch.grid(row=5, column=1, sticky='w')

# 输入num_steps
label_steps = tkinter.Label(root, text='num of steps:')
label_steps.grid(row=6, column=0, sticky='e')
text_steps = tkinter.IntVar()
text_steps.set(100)
spin_steps = tkinter.Spinbox(root, from_=100, to=160000, increment=100, textvariable=text_steps)
spin_steps.grid(row=6, column=1, sticky='w')


def is_number(s):
    """判段输入是否为一个数"""
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def compute_sigma(epsilon, delta):
    """compute sigma using gaussian mechanism"""
    return np.sqrt(2 * np.log(1.25 / delta)) / epsilon


# 开始训练并画图
def start_work():
    input_epsilon = entry_epsilon.get()
    input_delta = entry_delta.get()
    input_clip = entry_clip.get()
    if is_number(input_clip):
        if sigma_type.get():
            work_information.set('输入正确，运行中！')
        elif is_number(input_epsilon) and is_number(input_delta):
            work_information.set('输入正确，运行中！')
        else:
            work_information.set('输入格式错误，请重新输入！')
            return
    else:
        work_information.set('输入格式错误，请重新输入！')
        return

    sigma = 5
    if not sigma_type.get():
        epsilon = float(input_epsilon)
        delta = float(input_delta)
        sigma = compute_sigma(epsilon, delta)
    clip_bound = float(input_clip)
    batch_size = int(spin_batch.get())
    num_steps = int(spin_steps.get())
    target_delta = [0.00001, 0.0001, 0.001, 0.01]
    accuracy, loss = dp.main(sigma_type=sigma_type.get(), sigma=sigma, clip_bound=clip_bound,
                             batch_size=batch_size, num_steps=num_steps, target_delta=target_delta)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.plot(accuracy)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    delta_color = ['red', 'green', 'blue', 'black']
    for i in range(len(target_delta)):
        ax2.plot([epsilon_delta[i].spent_eps for epsilon_delta in loss],
                 label='delta=%.2g' % target_delta[i], color= delta_color[i])
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('epsilon')
    ax2.legend()

    plt.savefig('work1type%dsigma%fclip%fbatch%d.jpg' % (sigma_type.get(), sigma, clip_bound, batch_size))
    plt.show()


# 显示确定按钮
button = tkinter.Button(root, text='确定', command=start_work,
                        activeforeground='black', activebackground='blue', fg='white', bg='red')
button.grid(row=7, column=0, columnspan=2)

root.mainloop()
