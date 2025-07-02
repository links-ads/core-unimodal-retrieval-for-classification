from typing import Dict, Any, List, Union
import string
from warnings import warn
import random

class ClassifyQueryWithGenerator:
    """Class to classify queries using a HuggingFace generative model with retrieved examples."""
    
    def __init__(self, generator, example_template: str, prompt_template: str):
        """
        Initialize the classifier.
        
        Args:
            generator: The generator
            example_template: Template for formatting individual examples
            prompt_template: Template for the overall prompt
        """
        self.generator = generator
        self.example_template = example_template
        self.prompt_template = prompt_template
    
    def format_examples(self, 
                        sorted_docs: Dict[str, Dict[str, float]], 
                        corpus: Dict[str, Dict[str, Any]], 
                        ) -> str:
        """Format retrieved examples using the example template."""
        formatted_examples = []
        for i, (doc_id, score) in enumerate(sorted_docs.items()):
            if doc_id in corpus:
                doc_data = corpus[doc_id]
                # Extract text and label from the document data
                description = doc_data.get('text', '')
                label = "YES" if doc_data.get('label', False) else "NO"
                
                example = self.example_template.format(
                    i=i+1,
                    description=description,
                    label=label
                )
                formatted_examples.append(example)
            
        return "\n".join(formatted_examples)
    
    def build_prompt(self, query_text: str, examples: str) -> str:
        """Build the full prompt for the model."""
        return self.prompt_template.format(
            description=query_text,
            examples=examples
        )
    
    
    @staticmethod
    def _normalize(resp : str):
        _r = resp.strip().strip("\n").strip('.').upper()
        if "YES" in _r:
            return True
        if ( "NO" in _r) or ("NOT" in _r):
            return False
        else:
            # Split in words
            answer = _r.split()[0]
            answer = answer.strip().translate(str.maketrans('', '', string.punctuation)).upper()
            if answer == "YES":
                return True
            if (answer == "NO") or (answer == "NOT"):
                return False

        warn(f"The answer {resp} does not match neither YES nor NO.")
        return resp


    def classify(self, 
                similar_corpus: Dict[str, Dict[str, float]], 
                query: Dict[str, Dict[str, Any]], 
                corpus: Dict[str, Dict[str, Any]],
                k: List[int],
                print_prompt: bool = False
            ) -> Dict[str, Dict[int, Union[bool, Dict[str, Any]]]]:
        """
        Classify queries using the generator and retrieved examples.
        
        Args:
            similar_corpus: Dictionary mapping query IDs to dictionaries of doc_id -> score
            query: Dictionary mapping query IDs to query content
            corpus: Dictionary mapping doc IDs to document content
            k: List of top-k values to use for classification
            print_prompt: Whether to print the prompt for debugging
            
        Returns:
            Dictionary mapping query IDs to dictionaries of top-k -> classification results
        """
        results = {}
        
        for query_id in query:
            results[query_id] = {}
            query_text = query[query_id]['text']
            
            for top_k in k:
                # Filter similar docs based on top-k
                similar_docs = dict(sorted(similar_corpus[query_id].items(), 
                                        key=lambda item: item[1], reverse=True)[:top_k])
                # Format examples
                examples = self.format_examples(similar_docs, corpus)
                
                # Build the full prompt
                prompt = self.build_prompt(query_text, examples)
                # prompt = self.prompt_template.format(
                #     description=query_text,
                # )
                # Debug: print the prompt if requested
                if print_prompt:
                    print(f"=== PROMPT for query {query_id}, top-{top_k} ===")
                    print(prompt)
                    print("==============")
                
                # Generate response
                response = self.generator.generate(prompt)
                # Normalize the response
                response = self._normalize(response)
                # Store the result
                if response is not None:
                    results[query_id][top_k] = response
                    # append a file
                    f = open("/nfs/home/catalano/core-rag-benchmark/results1.txt", "a")
                    f.write(f"{query_id}, {response}")
                    f.close()
        # dict(qid: {k1: True/False, k2: True/False, ...})
        return results
