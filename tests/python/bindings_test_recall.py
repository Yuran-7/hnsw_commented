import os
import hnswlib
import numpy as np
import unittest

class RandomSelfTestCase(unittest.TestCase):
    def testRandomSelf(self):
        dim = 128
        num_elements = 20000
        k = 20
        num_queries = 10

        recall_threshold = 0.95

        # Generating sample data similar to SIFT1M format (whole number floats)
        data = np.random.randint(1, 100, (num_elements, dim)).astype(np.float32)

        # Declaring index
        hnsw_index = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip
        bf_index = hnswlib.BFIndex(space='l2', dim=dim)

        # Initializing both HNSW and brute force indices
        hnsw_index.init_index(max_elements=num_elements, ef_construction=200, M=16)
        bf_index.init_index(max_elements=num_elements)

        # Setting ef to control recall for HNSW
        hnsw_index.set_ef(200)

        # Optional: Set number of threads used during batch search/construction
        # hnsw_index.set_num_threads(4)

        print("Adding batch of %d elements" % (len(data)))
        hnsw_index.add_items(data)
        bf_index.add_items(data)

        print("Indices built")

        # Generating query data in similar format
        query_data = np.random.randint(1, 100, (num_queries, dim)).astype(np.float32)

        # Query the elements and measure recall
        labels_hnsw, distances_hnsw = hnsw_index.knn_query(query_data, k)
        labels_bf, distances_bf = bf_index.knn_query(query_data, k)

        # Measure recall
        correct = 0
        for i in range(num_queries):
            for label in labels_hnsw[i]:
                for correct_label in labels_bf[i]:
                    if label == correct_label:
                        correct += 1
                        break

        recall_before = float(correct) / (k * num_queries)
        print("Recall is:", recall_before)
        # self.assertGreater(recall_before, recall_threshold)

        # Test serialization of the brute force index
        # index_path = 'bf_index.bin'
        # print("Saving index to '%s'" % index_path)
        # bf_index.save_index(index_path)
        del bf_index

        # # Re-initiating and loading the index
        # bf_index = hnswlib.BFIndex(space='l2', dim=dim)
        # print("\nLoading index from '%s'\n" % index_path)
        # bf_index.load_index(index_path)

        # # Query the brute force index again to verify we get the same results
        # labels_bf, distances_bf = bf_index.knn_query(query_data, k)

        # # Measure recall after reloading
        # correct = 0
        # for i in range(num_queries):
        #     for label in labels_hnsw[i]:
        #         for correct_label in labels_bf[i]:
        #             if label == correct_label:
        #                 correct += 1
        #                 break

        # recall_after = float(correct) / (k * num_queries)
        # print("Recall after reloading is:", recall_after)

        # self.assertEqual(recall_before, recall_after)

        # os.remove(index_path)

if __name__ == '__main__':
    unittest.main()
