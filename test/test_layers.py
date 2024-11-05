import unittest
import torch
import torchformer


class TestTransformerEncoderBlock(unittest.TestCase):

    def test_forward(self):
        d_model = 512
        BATCH_SIZE = 10

        model = torchformer.TransformerEncoderBlock(d_model=d_model)
        data = torch.randn(BATCH_SIZE, d_model)
        output = model(data)
        self.assertEqual(output.shape, (BATCH_SIZE, d_model))


if __name__ == '__main__':
    unittest.main()
