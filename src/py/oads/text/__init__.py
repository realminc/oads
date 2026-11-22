"""Text datasets: character and document corpora."""

from .ag_news import AgNews
from .imdb import Imdb
from .tiny_shakespeare import TinyShakespeare
from .yelp_polarity import YelpPolarity

__all__ = ["AgNews", "Imdb", "TinyShakespeare", "YelpPolarity"]
