from unittest import TestCase

import numpy
from phi.tf import tf

from phi.tf.util import placeholder

from phi import math, struct
from phi.geom import box
from phi.physics.field import CenteredGrid
from phi.tf.util import variable


class TestPlaceholder(TestCase):

    def test_direct_placeholders(self):
        tf.reset_default_graph()
        p = placeholder([4])
        self.assertIsInstance(p, tf.Tensor)
        numpy.testing.assert_equal(p.shape.as_list(), [4])
        self.assertEqual(p.name, 'Placeholder:0')
        v = variable(math.zeros([2, 2]))
        numpy.testing.assert_equal(v.shape.as_list(), [2, 2])
        self.assertIsInstance(v, tf.Variable)
        self.assertEqual(v.name, 'Variable:0')

    def test_struct_placeholders(self):
        obj = ([4], CenteredGrid([1, 4, 1], box[0:1], content_type=struct.shape), ([9], [8, 2]))
        tf.reset_default_graph()
        p = placeholder(obj)
        self.assertEqual('Placeholder/0:0', p[0].name)
        self.assertEqual('Placeholder/1/data:0', p[1].data.name)
        self.assertIsInstance(p, tuple)
