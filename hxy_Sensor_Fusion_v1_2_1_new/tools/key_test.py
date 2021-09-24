from pynput.keyboard import Key, Controller,Listener
keyboard = Controller()
def on_press(key):
    print('{0} 被按下'.format(key))


def on_release(key):
    print('{0} 被释放'.format(key))
    if key == Key.esc:
        return False
    if key not in Key and key.char == 'M':# ctrl 0
        test()

def test():
	print ('按下ctrl 0,运行测试程序')
# 创建监听
with Listener(on_press=on_press,on_release=on_release) as listener:
    listener.join()