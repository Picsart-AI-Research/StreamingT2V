# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
# Copyright (c) Alibaba, Inc. and its affiliates.
from modelscope.utils.registry import Registry, build_from_cfg, default_group

HOOKS = Registry('hooks')


def build_hook(cfg, default_args=None):
    return build_from_cfg(
        cfg, HOOKS, group_key=default_group, default_args=default_args)
