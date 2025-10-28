# %% [markdown]

# %%
import datetime
import re

def shift_srt_forward(filename, delta):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # will commit to me?

    shifted_lines = []

    for line in lines:
        match = re.search('(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', line)
        if match:
            start_time_str, end_time_str = match.groups()
            start_time = datetime.datetime.strptime(start_time_str, '%H:%M:%S,%f')
            end_time = datetime.datetime.strptime(end_time_str, '%H:%M:%S,%f')

            # Subtraia o deslocamento
            start_time -= datetime.timedelta(seconds=delta)
            end_time -= datetime.timedelta(seconds=delta)

            line = line.replace(start_time_str, start_time.strftime('%H:%M:%S,%f')[:-3])
            line = line.replace(end_time_str, end_time.strftime('%H:%M:%S,%f')[:-3])

        shifted_lines.append(line)

    with open(filename, 'w') as file:
        file.writelines(shifted_lines)


def shift_srt_backward(filename, delta):
    with open(filename, 'r') as file:
        lines = file.readlines()

    shifted_lines = []

    for line in lines:
        match = re.search('(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', line)
        if match:
            start_time_str, end_time_str = match.groups()
            start_time = datetime.datetime.strptime(start_time_str, '%H:%M:%S,%f')
            end_time = datetime.datetime.strptime(end_time_str, '%H:%M:%S,%f')

            # Adicione o deslocamento
            start_time += datetime.timedelta(seconds=delta)
            end_time += datetime.timedelta(seconds=delta)

            line = line.replace(start_time_str, start_time.strftime('%H:%M:%S,%f')[:-3])
            line = line.replace(end_time_str, end_time.strftime('%H:%M:%S,%f')[:-3])

        shifted_lines.append(line)

    with open(filename, 'w') as file:
        file.writelines(shifted_lines)

# just add the srt file and set the time u want forward or backward in seconds
shift_srt_forward('/content/legenda.srt', 1)
print("Finished")


# %%



