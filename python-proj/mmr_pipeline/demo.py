import querying.query_app as qr
import scalability.clustering as cluster
import matplotlib.pyplot as plt
import numpy as np

plot_colors, x, y, shape_classes, file_names = cluster.apply_tsne()

fig, ax = plt.subplots()

names = shape_classes
x = x
y = y
colors = plot_colors
#sizes = [20*4**n for n in range(len(x))]

tolerance = 3 # points

sc = ax.scatter(x=x, y=y, color=colors, picker=tolerance, label=[names], s=16.0)

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"), fontsize = 15)
annot.set_visible(False)


def on_pick(event):
    artist = event.artist
    xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
    ind = event.ind
    print(file_names[ind[0]])
    qr.query_from_tsne(file_names[ind[0]])

def update_annot(ind):
    
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    #text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), " ".join([names[n] for n in ind["ind"]]))
    #index = int(ind[0])
    index = ind["ind"][0]
    text = names[index]
    annot.set_text(text)
    #annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
    annot.get_bbox_patch().set_alpha(0.8)
    

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)
fig.canvas.callbacks.connect('pick_event', on_pick)

# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#           fancybox=True, shadow=True, ncol=2)

plt.show()