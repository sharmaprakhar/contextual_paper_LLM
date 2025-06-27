import os
import json
import prompts
from typing import List, Dict, Any, Tuple

class PaperLoader:
    def __init__(self, paper_dir, paper, model) -> None:
        self.paper_dir = paper_dir
        paper_json = os.path.join(self.paper_dir, paper)
        self.paper = paper_json
        self.model = model
        self.load_paper()

    def load_paper(self) -> Tuple[List[Dict[str, Any]], str]:
        with open(self.paper, 'r') as fh:
            data = json.load(fh)
        self.paper_structure = ''
        self.content = data['body']
        for doc in self.content:
            self.paper_structure += doc['title']
        
        # print('curent paper structure:', self.paper_structure)
        
        abstract_done = False

        # write a test for  the following loop
        for doc in self.content:
            text = ''.join(doc['content'])
            if not abstract_done:
                if 'abstract' in doc['title'].lower() or 'abstract' in text.lower():
                    print('found abstract in:', doc['title'])
                    input = prompts.summarize_abstract_prompt.format(doc['content'])
                    self.abstract_summary = self.model.invoke(input)
                    abstract_done = True
        return
    
if __name__=="__main__":
    ploader = PaperLoader("paper_jsons/5Greasoner.json", "model")
    