import sys

# Flush progress to the command line, progress value should be between 0 to 1
def progress_bar(progress, bar_length=60):
    block = int(round(bar_length * progress))
    text = "\r|{0}| {1:.0f}%".format("█" * block + " " * (bar_length - block), progress * 100)
    sys.stdout.write(text)
    sys.stdout.flush()