"""
Pool Keeper Package
===================
Moteur de projections et d'analyse pour Pool GOOD YEAR Keeper (LNH).
"""

from pool_keeper.config import (
    BENCH_DEFAULT,
    COLCFG,
    DEFAULT_N_SEASONS,
    DEFAULT_N_TEAMS,
    GAMES,
    KEEPERS,
    MAX_YEARS,
    METHODO,
    RULES,
    SLOTS,
)
from pool_keeper.data import cache_purge_nonfinal, current_season_id, load_data
from pool_keeper.model import apply_weights, build_model
from pool_keeper.trade import compare_trade, summarize_roster
from pool_keeper.trade_finder import find_target, suggest_offers
from pool_keeper.ui import detect_mobile_mode
from pool_keeper.utils import match_names, norm, parse_names, season_label, season_shift, to_view

__all__ = [
    "RULES",
    "SLOTS",
    "BENCH_DEFAULT",
    "DEFAULT_N_SEASONS",
    "DEFAULT_N_TEAMS",
    "KEEPERS",
    "GAMES",
    "MAX_YEARS",
    "METHODO",
    "COLCFG",
    "cache_purge_nonfinal",
    "current_season_id",
    "load_data",
    "build_model",
    "apply_weights",
    "compare_trade",
    "summarize_roster",
    "suggest_offers",
    "find_target",
    "detect_mobile_mode",
    "norm",
    "season_shift",
    "season_label",
    "parse_names",
    "match_names",
    "to_view",
]

