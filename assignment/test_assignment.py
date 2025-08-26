"""Testing the function dicePosterior, which calculates the posterior
probability of selecting each die type in the bag-of-dice problem."""

import logging
import unittest
from gradescope_utils.autograder_utils.decorators import weight
# Handle both VS Code (relative import) and autograder (absolute import) contexts
try:
    from .assignment import oo_generate_sample, Die, dice_posterior  # VS Code context
except ImportError:
    from assignment import oo_generate_sample, Die, dice_posterior  # Autograder context

#configure_logging(logging.DEBUG)

class TestObjectOrientedDiceSample(unittest.TestCase):
  @weight(0)
  def test_Die(self):
     d = Die([.1, .9])
     self.assertEqual(d.roll(3).shape, (3,))
  
  @weight(5)
  def test_sampled_value_range(self):
      """Testing that the sampled values lie in the correct range and that 
      0 probabilities are handled correctly."""
      sample = oo_generate_sample((1, 0), 
                                  (Die((1/3, 1/3, 1/3)), Die((1/5, 1/5, 1/5, 1/5, 1/5))), 
                                 7, 10)
      min_val = min((min(x) for x in sample))
      max_val = max((max(x) for x in sample))
      self.assertGreaterEqual(min_val, 0)
      self.assertLessEqual(max_val, 2)

class TestDicePosterior(unittest.TestCase):
    @weight(6)
    def test_case_1(self):
        """Testing that the dimensions of the sample are correct."""

        sample_draw = [1, 1, 1, 1]

        # while [1/4]*4 may look like it should return 1, it is a short hand
        # for "repeat". The result of [1/4] * 4 = [1/4, 1/4, 1/4, 1/4]
        result = dice_posterior(sample_draw, [1, 1], [Die([1/4]*4), Die([1/4]*4)])
        self.assertEqual(result, 0.5)

    @weight(6)
    def test_case_2(self):
        """Testing that, when the counts provide no evidence favoring either
          die type, the posterior is equal to the prior."""

        sample_draw = [1, 1, 1, 1]
        result = dice_posterior(sample_draw, 
                                [1, 2], 
                                [Die([1/4]*4), Die([1/4]*4)])
        self.assertAlmostEqual(result, 0.33, places=2)

    @weight(4)
    def test_case_3(self):
        """Testing that the likelihood can overcome the prior."""

        sample_draw = [1, 1, 1, 4]
        result = dice_posterior(sample_draw,
                                [1, 2],
                                [Die([1/8, 1/8, 1/8, 5/8]), Die([1/4]*4)])
        self.assertAlmostEqual(result, 0.71, places=2)

    @weight(4)
    def test_case_4(self):
        """Testing the handling of a zero in the counts but not in the
          probabilities."""

        sample_draw = [1, 0, 1, 4]
        result = dice_posterior(sample_draw, 
                                [1, 2],
                                [Die([1/8, 1/8, 1/8, 5/8]),
                                 Die([1/4]*4)])
        self.assertAlmostEqual(result, 0.83, places=2)

    @weight(4)
    def test_case_5(self):
        """Testing the handling of a zero in the probabilities but not in the
          counts."""

        sample_draw = [1, 1, 1, 4]
        result = dice_posterior(sample_draw, 
                                [1, 2],
                                [Die([0., 1/8, 1/8, 6/8]),
                                 Die([1/4]*4)])
        self.assertEqual(result, 0)

    @weight(4)
    def test_case_6(self):
        """Testing the handling of a zero in both the probabilities and the
          counts."""

        sample_draw = [0, 1, 1, 4]
        result = dice_posterior(sample_draw, 
                                [1, 2],
                                [Die([0., 1/8, 1/8, 6/8]),
                                 Die([1/4]*4)])
        self.assertAlmostEqual(result, 0.91, places=2)

    @weight(4)
    def test_case_7(self):
        """Testing the handling of a zero in both the probabilities and the
          counts."""

        sample_draw = [0, 1, 1, 4]
        result = dice_posterior(sample_draw, 
                                [1/3, 2/3],
                                [Die([0., 1/8, 1/8, 6/8]),
                                 Die([1/4, 0., 1/4, 2/4])])
        self.assertAlmostEqual(result, 1.0, places=2)

    @weight(4)
    def test_case_8(self):
        """Testing the handling of a zero in the priors, counts, and face
          probs."""
        
        sample_draw = [0, 1, 1, 4]
        result = dice_posterior(sample_draw, 
                                [0., 1.],
                                [Die([0., 1/8, 1/8, 6/8]),
                                 Die([1/4]*4)])
        self.assertEqual(result, 0)

    @weight(4)
    def test_case_9(self):
        """Testing that the code properly infers the number of faces on the
          dice from the probablity vectors."""

        sample_draw = [1, 1, 1, 4, 1, 1, 1]
        result = dice_posterior(sample_draw, 
                                [1/3, 2/3],
                                [Die([0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1]),
                                 Die([1/7]*7)])
        self.assertAlmostEqual(result, 0.78, places=2)
