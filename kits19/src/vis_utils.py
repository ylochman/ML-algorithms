import matplotlib.pyplot as plt

def show_slices(overlayed_volume, columns=3, figsize=(50, 50)):
    """ Function to display row of image slices """
    rows = (overlayed_volume.shape[0] // columns) + 1
    fig, axes = plt.subplots(rows, columns, figsize=figsize)
    for i in range(overlayed_volume.shape[0]):
        row = i // columns
        column = i - row * columns 
        slice = overlayed_volume[i, :, :, :]
        axes[row][column].imshow(slice)

def multi_slice_viewer(volume, first_index=0):
    """ Function to display image slices with ability to scroll them accross the depth
        > press K to view next slice
        > press J to view previous slice
    """
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = first_index#volume.shape[0] // 2
    ax.imshow(volume[ax.index])
    ax.set_title('slice {}'.format(ax.index))
    fig.canvas.mpl_connect('key_press_event', process_key)
        
def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])
    ax.set_title('slice {}'.format(ax.index))

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])
    ax.set_title('slice {}'.format(ax.index))