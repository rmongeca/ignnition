import ignnition
import pytest
from tests.utils import working_directory

example_dir = "examples/Routenet"

@pytest.mark.timeout(60*2)  # Timeout after 2 minutes
def test_routenet():
    """Test Routenet example"""
    with working_directory(example_dir):
        model = ignnition.create_model(model_dir= './')
        # Set epochs to a limited amount for time constraints
        model.CONFIG["epochs"] = 1
        model.computational_graph()
        model.train_and_validate()
