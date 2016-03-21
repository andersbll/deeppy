from ... import debug_mode
from .exprgraph import ExprGraph
from .util import draw, DebugExprGraph, NANGuardExprGraph

if debug_mode == 'trace':
    ExprGraph = DebugExprGraph

if debug_mode == 'nan':
    ExprGraph = NANGuardExprGraph
