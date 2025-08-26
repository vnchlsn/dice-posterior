import logging
from typing import List, Union, Tuple, Optional
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# First, reimplement dice-sample using the Die class defined below.
# Previously, we passed in the probability of the faces of a die as a tuple and
# defined a function "roll" to roll the die. An alternative that is more in
# typical python idiom is to define a class for the die and have a method to roll
# the die. A minimalist version of this is shown below.

class Die:
    def __init__(self, face_probs: Tuple[float]):
        self._face_probs = face_probs
    
    @property
    def face_probs(self) -> Tuple[float]:
        return self._face_probs
    
    @face_probs.setter
    def face_probs(self, face_probs: Tuple[float]):
        self._face_probs = face_probs

    @property
    def num_faces(self) -> int:
        return len(self._face_probs) 
    
    def roll(self, num_rolls: int, seed: Optional[int] = 42) -> NDArray[np.integer]:
        np.random.seed(seed)  # Set the seed for reproducibility
        return np.random.choice(range(self.num_faces), num_rolls, p=self._face_probs)

# Defining this class creates a new type, Die, for type hints.
# Now we can implement an object oriented generate_sample using this class.

def oo_generate_sample(die_type_counts: Tuple[int],
               dice: Tuple[Die],
               num_draws: int,
               rolls_per_draw: int) -> NDArray[np.integer]:
    """
    Args:
        die_type_counts (Tuple[int]): The number of each type of die present
        in the bag. 
        dice (Tuple[Die]): A tuple specifying the die types in the bag.         
        num_draws (int): The number of times to pull a die from the bag
        rolls_per_draw (int): The number of times each selected die is rolled.

    Returns:
        NDArray[np.integer]: A numpy array of draws, each draw being a tuple 
        of faces rolled.
    """
    die_type_counts_array = np.array(die_type_counts)
    die_type_probs = die_type_counts_array / sum(die_type_counts_array)

    
    # A a numpy ndarray of indices of randomly selected dice 
    die_types_drawn = np.random.choice(len(die_type_probs), 
                                       num_draws, 
                                       p= die_type_probs)
    # Iterate through die_types_drawn and use each die type index to obtain
    # the corresponding die. Then use its roll method to roll it rolls_per_draw
    # times. Return the resulting np.ndarray.

    # PUT YOUR CODE HERE. Make sure to use the "roll" method from Die and to 
    # return a numpy nd.array.


def safe_exponentiate(base: Union[int, float],
                      exponent: Union[int, float]) -> Union[int, float]:
    """Exponentiates while handling the 0^0 in a way that is appropriate
    for calculating likelihood in the dice example.
    """
    # YOUR CODE HERE


def dice_posterior(sample_draw: List[int], 
                   die_type_counts: Tuple[int],
                   dice: Tuple[Die]) -> float:
    """Calculates the posterior probability of a type 1 vs a type 2 die,
    based on the number of times each face appears in the draw, and the
    relative numbers of type 1 and type 2 dice in the bag, as well as the
    face probabilities for type 1 and type 2 dice. The single number returned
    is the posterior probability of the Type 1 die. Note: we expect a BagOfDice
    object with only two dice.

    """
    # Requiring only two dice with the same number of faces simplifies the
    # problem a bit.
    if len(dice) != 2:
        raise ValueError('This code requires exactly 2 dice')
    if dice[0].num_faces != dice[1].num_faces:
        raise ValueError('This code requires two dice with the same number of faces')
    if len(sample_draw) != dice[0].num_faces:
        raise ValueError('The sample draw is a list of observed counts for the \
                         faces. Its length must be equal to the number of faces \
                         on the dice.')
    # YOUR CODE HERE. You may want to use your safe_exponeniate.
