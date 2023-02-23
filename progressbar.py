from datetime import datetime
import time
# Print iterations progress
class PrintProgressBar:

    def __init__(self, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.printEnd = printEnd
        self.start = self.__get_time()

    def __s2h(self, seconds):
        # print(seconds)
        hours = seconds // 3600
        seconds = seconds - hours * 3600
        # print(seconds)
        minutes = seconds // 60
        seconds = seconds - 60 * minutes
        return (hours, minutes, seconds)


    def __get_time(self):
        return time.time()

    def printProgressBar(self, iteration):
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (iteration / float(self.total)))
        filledLength = int(self.length * iteration // self.total)
        bar = self.fill * filledLength + '-' * (self.length - filledLength)
        current_time = self.__get_time()
        seconds = current_time - self.start
        __time = self.__s2h(seconds)
        cost_time = f"{__time[0]:>.0f}:{__time[1]:0>2.0f}:{__time[2]:0>2.0f}"
        ## 计算剩余时间
        try:
            seconds1 = seconds * float(self.total) / iteration - seconds
            __time = self.__s2h(seconds1)
        except ZeroDivisionError:
            __time = [0, 0, 0]
        left_time = f"{__time[0]:>.0f}:{__time[1]:0>2.0f}:{__time[2]:0>2.0f}"

        print('\r%s |%s| %s%% %s %s|%s' % (self.prefix, bar, percent, self.suffix, cost_time, left_time), end=self.printEnd)
        # Print New Line on Complete
        if iteration == self.total:
            print()


def printProgressBar(self, iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


if __name__ == "__main__":
    pass
