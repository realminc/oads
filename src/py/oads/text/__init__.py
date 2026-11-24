"""Text datasets: character and document corpora."""

from .ag_news import AgNews
from .anli_r1 import AnliR1
from .anli_r2 import AnliR2
from .anli_r3 import AnliR3
from .arc_challenge import ArcChallenge
from .arc_easy import ArcEasy
from .cfq import Cfq
from .dbpedia import Dbpedia
from .enwik8 import Enwik8
from .flores101 import Flores101
from .glue_cola import GlueCola
from .glue_mnli import GlueMnli
from .glue_qnli import GlueQnli
from .glue_qqp import GlueQqp
from .glue_rte import GlueRte
from .glue_sst2 import GlueSst2
from .glue_stsb import GlueStsb
from .glue_wnli import GlueWnli
from .go_emotions import GoEmotions
from .imdb import Imdb
from .lm1b import Lm1b
from .math_dataset import MathDataset
from .multi_nli import MultiNli
from .openbookqa import Openbookqa
from .piqa import Piqa
from .scan import Scan
from .squad_v1 import SquadV1
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
from .trivia_qa import TriviaQa
from .wiki40b import Wiki40B
from .winogrande import Winogrande
from .xnli import Xnli
from .xsum import Xsum
from .yelp_polarity import YelpPolarity
from .yelp_review_full import YelpReviewFull

__all__ = ["AgNews", "AnliR1", "AnliR2", "AnliR3", "ArcChallenge", "ArcEasy", "Cfq", "Dbpedia", "Enwik8", "Flores101", "GlueCola", "GlueMnli", "GlueQnli", "GlueQqp", "GlueRte", "GlueSst2", "GlueStsb", "GlueWnli", "GoEmotions", "Imdb", "Lm1b", "MathDataset", "MultiNli", "Openbookqa", "Piqa", "Scan", "SquadV1", "SquadV2", "SuperglueBoolq", "SuperglueCb", "SuperglueCopa", "SuperglueMultirc", "SuperglueRecord", "SuperglueRte", "SuperglueWic", "SuperglueWsc", "TinyShakespeare", "TriviaQa", "Wiki40B", "Winogrande", "Xnli", "Xsum", "YelpPolarity", "YelpReviewFull"]
