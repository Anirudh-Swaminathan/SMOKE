import sys
from importlib import util

import logging


# from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
def import_file(module_name, file_path, make_importable=False):
    l = logging.getLogger(__name__)
    l.info("IN import_file({}, {}, {})".format(module_name, file_path, make_importable))
    spec = util.spec_from_file_location(module_name, file_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if make_importable:
        sys.modules[module_name] = module_name
    return module