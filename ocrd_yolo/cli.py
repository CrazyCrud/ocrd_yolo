import click
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from .segment import Yolo2Segment

@click.command()
@ocrd_cli_options
def ocrd_yolo_segment(*args, **kwargs):
    return ocrd_cli_wrap_processor(Yolo2Segment, *args, **kwargs)