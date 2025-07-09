import unittest
from unittest.mock import patch, MagicMock
import os

# Import functions from main.py
from main import (
    initialize_constants,
    add_metadata,
    create_knowledge_base,
    create_vectorstore,
    create_conversation_chain,
    chat
)

class TestMain(unittest.TestCase):
    @patch('rag_electric.main.load_dotenv')
    @patch('rag_electric.main.os')
    def test_initialize_constants(self, mock_os, mock_load_dotenv):
        mock_os.getenv.return_value = 'test-key'
        initialize_constants()
        self.assertEqual(mock_os.environ['OPENAI_API_KEY'], 'test-key')

    def test_add_metadata(self):
        doc = MagicMock()
        doc.metadata = {}
        doc_type = 'test_type'
        result = add_metadata(doc, doc_type)
        self.assertEqual(result.metadata['doc_type'], doc_type)

    @patch('rag_electric.main.glob.glob')
    @patch('rag_electric.main.DirectoryLoader')
    @patch('rag_electric.main.CharacterTextSplitter')
    def test_create_knowledge_base(self, mock_splitter, mock_loader, mock_glob):
        mock_glob.return_value = ['folder1']
        mock_loader.return_value.load.return_value = [MagicMock(metadata={})]
        mock_splitter.return_value.split_documents.return_value = ['chunk1', 'chunk2']
        chunks = create_knowledge_base()
        self.assertEqual(chunks, ['chunk1', 'chunk2'])

    @patch('rag_electric.main.OpenAIEmbeddings')
    @patch('rag_electric.main.os.path.exists')
    @patch('rag_electric.main.Chroma')
    def test_create_vectorstore(self, mock_chroma, mock_exists, mock_embeddings):
        mock_exists.return_value = False
        mock_chroma.from_documents.return_value._collection.count.return_value = 2
        mock_chroma.from_documents.return_value._collection.get.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_chroma.from_documents.return_value._collection.count.return_value = 1
        chunks = ['chunk1']
        vectorstore = create_vectorstore(chunks)
        self.assertTrue(mock_chroma.from_documents.called)
        self.assertIsNotNone(vectorstore)

    @patch('rag_electric.main.ChatOpenAI')
    @patch('rag_electric.main.ConversationBufferMemory')
    @patch('rag_electric.main.ConversationalRetrievalChain')
    def test_create_conversation_chain(self, mock_chain, mock_memory, mock_llm):
        vectorstore = MagicMock()
        vectorstore.as_retriever.return_value = 'retriever'
        mock_chain.from_llm.return_value = 'conversation_chain'
        result = create_conversation_chain(vectorstore)
        self.assertEqual(result, 'conversation_chain')

    def test_chat(self):
        conversation_chain = MagicMock()
        conversation_chain.invoke.return_value = {"answer": "test answer"}
        result = chat("question", [], conversation_chain)
        self.assertEqual(result, "test answer")

if __name__ == '__main__':
    unittest.main() 