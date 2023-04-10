import unittest
from dht import DistributedHashTable

"""
Tester class for the distributed hash table.
"""
class TestDHT(unittest.TestCase):

    """
    Simple set-up procedure. This function is executed before any of the tests
    do.
    """
    def setUp(self):
        print("Initializing hash table...")
        self.dht = DistributedHashTable()

    """
    Test whether a single combination of `put` and `get` works as expected.
    """
    def testSinglePutGet(self):
        # Dictionary we will compare against.
        dummy_dictionary = {
            'ip': None,
            'port': None,
            'location': None,
            'status': None,
            'stages': None,
            'load': None
        }
        # Create dummy entry for server 0.
        self.dht.put((0, None), None)
        # Assert those two are equal.
        self.assertEqual(self.dht.get((0, None)), dummy_dictionary)

unittest.main()
