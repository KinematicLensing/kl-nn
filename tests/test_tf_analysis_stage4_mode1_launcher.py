from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "arch" / "tf_analysis_stage4_mode1_10k.slurm"


def test_stage4_mode1_launcher_is_paired_and_reproducible():
    text = LAUNCHER.read_text()

    assert "#SBATCH --array=1-10" in text
    assert 'MODEL_NAME="${MODEL_NAME:-CNN-SetAttn-D4-bounded-hybrid-circular-flow_tf_stage6_fixedfiber_s42_43747436}"' in text
    assert 'EPOCH="${EPOCH:-19}"' in text
    assert 'SEED="${SEED:-42}"' in text
    assert 'NETWORK_SOURCE="${NETWORK_SOURCE:-current}"' in text
    assert 'NGALS="${NGALS:-1000}"' in text
    assert 'NSAMPLES="${NSAMPLES:-5000}"' in text
    assert 'NPARTS="${NPARTS:-10}"' in text
    assert 'CACHE_TAG="${CACHE_TAG:-d4_exact_mode1_raw_10k_s${SEED}}"' in text

    assert "NGALS=1000 per partition (10,000 total)" in text

    assert "--mode 1" in text
    assert "--conform-to-tf" in text
    assert "--no-cancel-add-noise" in text
    assert "--no-compile" in text
    assert "--no-amp" in text
    assert "--channels-last" in text
    assert "--inference-mode" in text
    assert "--seed " in text
    assert '--network-source "${NETWORK_SOURCE}"' in text

    assert 5000 % 8 == 0
    assert 1000 * 10 == 10000
