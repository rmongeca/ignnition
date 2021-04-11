import ignnition
import pytest
from tests.utils import working_directory

example_dir = "examples/Shortest_Path"

@pytest.mark.timeout(60*5)  # Timeout after 5 minutes
def test_shortest_path():
    """Test Shortest Path example"""
    with working_directory(example_dir):
        model = ignnition.create_model(model_dir= './')
        # Set epochs to a limited amount for time constraints
        model.CONFIG["epochs"] = 1
        model.computational_graph()
        model.train_and_validate()