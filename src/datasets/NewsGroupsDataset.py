from sklearn.datasets import fetch_20newsgroups

class NewsGroupsDataset:
    def load(self, categories=None, ids_db=None, ids_query=None):
        # Load the full dataset (both train and test)
        data_train = fetch_20newsgroups(subset='train', categories=categories, 
                                        remove=('headers', 'footers', 'quotes'))
        data_test = fetch_20newsgroups(subset='test', categories=categories,
                                      remove=('headers', 'footers', 'quotes'))
        
        # Combine train and test
        texts = data_train.data + data_test.data
        labels = list(data_train.target) + list(data_test.target)
        categories = data_train.target_names
        
        # Create IDs for all documents
        all_ids = [f"doc_{i}" for i in range(len(texts))]
        
        # Use provided split IDs or create a default split
        if ids_db is not None and ids_query is not None:
            db_indices = [all_ids.index(id) for id in ids_db if id in all_ids]
            query_indices = [all_ids.index(id) for id in ids_query if id in all_ids]
        else:
            # Use all IDs if none specified
            db_indices = list(range(len(all_ids)))
            query_indices = list(range(len(all_ids)))
        
        # Populate corpus with documents
        self.corpus = {
            all_ids[i]: {
                'text': texts[i],
                'labels': categories[labels[i]]
            } for i in db_indices
        }
        
        # Populate queries
        self.queries = {
            all_ids[i]: {
                'text': texts[i],
                'labels': categories[labels[i]]
            } for i in query_indices
        }
        
        self.all_labels = categories
        return self
