
import sys
import subprocess

def upload(pat,dest):
    """ Upload objects matched to a given pattern.
    Args:
      pat: Pattern of objects to be uploaded.
      dest: Destination URL.
    """

    proc = subprocess.Popen(
        ["gsutil", "-m", "cp", pat, dest], stdout=sys.stdout, stderr=sys.stderr)
    proc.wait()

if __name__ == "__main__":
    pass
