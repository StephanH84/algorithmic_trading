import matplotlib.pyplot as plt
import time
from datetime import datetime
plt.style.use('ggplot')

def get_datetime(str_, format="%d.%m.%Y"):
    result = datetime.strptime(str_, format)
    return result

def plot_data(data, limit=1000, name="output"):
    classic_dashes = {
        "lines.markersize": 3
    }
    with plt.rc_context(classic_dashes):
        for point in data[:limit]:
            plt.plot(get_datetime(point[0]),point[1], 'r.')

        datetime_string = str(datetime.now())[0:19]
        datetime_string = datetime_string.replace(' ', '-').replace(':','-')
        plt.savefig("../plots/%s-%s.png" % (name, datetime_string))

        plt.show()