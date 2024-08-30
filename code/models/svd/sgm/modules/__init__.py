from models.svd.sgm.modules.encoders.modules import GeneralConditioner

UNCONDITIONAL_CONFIG = {
    "target": "sgm.modules.GeneralConditioner",
    "params": {"emb_models": []},
}
