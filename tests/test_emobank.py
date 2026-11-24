"""EmoBank loader: genres from meta.tsv, VA rescaled from [1,5] to [-1,1]."""
from eval import emobank


def test_load_corpus(tmp_path):
    (tmp_path / "emobank.csv").write_text(
        'id,split,V,A,D,text\n'
        'a1,train,5.0,1.0,3.0,"Great, ""really"" great."\n'
        'b2,dev,3.0,3.0,3.0,Plain.\n'
        'c3,test,1.0,5.0,3.0,Awful.\n'
        'd4,test,3.0,3.0,3.0,Unknown genre.\n')
    (tmp_path / "meta.tsv").write_text(
        "id\tdocument\tcategory\tsubcategory\n"
        "a1\tx\tblog\t \nb2\tx\tSemEval\t \nc3\tx\ttravel-guides\t \nd4\tx\tpoetry\t \n")
    text, y, genre, split = emobank.load_corpus(tmp_path)
    assert len(text) == 3                                  # unknown genre dropped
    assert y.tolist() == [[1.0, -1.0], [0.0, 0.0], [-1.0, 1.0]]
    assert [emobank.GENRES[g] for g in genre] == ["blog", "SemEval", "travel-guides"]
    assert list(split) == ["train", "dev", "test"]
