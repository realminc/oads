"""Text datasets: character and document corpora."""

from .ag_news import AgNews
from .anli_r1 import AnliR1
from .anli_r2 import AnliR2
from .anli_r3 import AnliR3
from .glue_cola import GlueCola
from .glue_mnli import GlueMnli
from .glue_qnli import GlueQnli
from .glue_qqp import GlueQqp
from .glue_rte import GlueRte
from .glue_sst2 import GlueSst2
from .glue_stsb import GlueStsb
from .glue_wnli import GlueWnli
from .imdb import Imdb
from .squad_v2 import SquadV2
from .superglue_boolq import SuperglueBoolq
from .superglue_cb import SuperglueCb
from .superglue_copa import SuperglueCopa
from .superglue_multirc import SuperglueMultirc
from .superglue_record import SuperglueRecord
from .superglue_rte import SuperglueRte
from .superglue_wic import SuperglueWic
from .superglue_wsc import SuperglueWsc
from .tiny_shakespeare import TinyShakespeare
from .xnli import Xnli
from .yelp_polarity import YelpPolarity

__all__ = ["AgNews", "AnliR1", "AnliR2", "AnliR3", "GlueCola", "GlueMnli", "GlueQnli", "GlueQqp", "GlueRte", "GlueSst2", "GlueStsb", "GlueWnli", "Imdb", "SquadV2", "SuperglueBoolq", "SuperglueCb", "SuperglueCopa", "SuperglueMultirc", "SuperglueRecord", "SuperglueRte", "SuperglueWic", "SuperglueWsc", "TinyShakespeare", "Xnli", "YelpPolarity"]
