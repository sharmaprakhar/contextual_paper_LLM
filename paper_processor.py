import os
import prompts
from llama_index.core import Document

class PaperProcessor:
    def __init__(self, embed_model, model, paper) -> None:
        self.embed_model = embed_model 
        self.model = model
        self.paper = paper

    def is_heading(self, title: str) -> bool:
        input = prompts.is_heading_prompt.format(self.paper.paper_structure, title)
        sample_out = self.model.invoke(input)
        if 'yes' in sample_out.lower():
            return True
        return False
    
    def process_heading(self, title: str,  text: str) -> None:
        input = prompts.process_heading_prompt.format(self.paper.paper_structure, self.paper.abstract_summary, text)
        self.current_heading_summary = self.model.invoke(input)
        self.current_title = title
        

    def process_subheading(self, text: str) -> str:
        current_heading = self.current_title + self.current_heading_summary
        input = prompts.process_subheading_prompt.format(self.paper.paper_structure, self.paper.abstract_summary, current_heading, text)
        response = self.model.invoke(input)
        return response
    
    def get_semantic_documents(self) -> None:
        documents = []
        for section in self.paper.content:
            title = section['title']
            # remove references and conclusion
            if 'reference' in title.lower() or 'conclusion' in title.lower():
                continue
            text = section['content']
            is_heading = self.is_heading(title)
            if is_heading:
                print('\nProcessing section: {} as a heading'.format(title))
                self.process_heading(title, text)
                # create document
                documents.append(Document(
                    text = self.current_heading_summary,
                    metadata = {
                        "text_type" : "heading",
                        "title" : title
                    }
                ))
            else:
                print('\nProcessing section: {} as a subheading'.format(title))
                subheading_summary = self.process_subheading(text)
                documents.append(Document(
                    text = subheading_summary,
                    metadata = {
                        "text_type" : "subheading",
                        "main_heading" : self.current_title,
                        "title" : title
                    }
                ))
        return documents