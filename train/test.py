class test:
    def __init__(self,time):
        self.time = time


t = test(10)
a = t
a.time  =5
print(t.time)