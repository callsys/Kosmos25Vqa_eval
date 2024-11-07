import importlib
import io
import os
import os.path as osp

import fsspec
from adlfs import AzureBlobFileSystem
from azure.storage.blob import BlobServiceClient


def is_detectron2__available():
    return importlib.util.find_spec("detectron2") is not None


def is_amlk8s():
    return "AMLK8S_NUM_WORKER" in os.environ or "AZ_CMK8S_JOB_WORK_DIR" in os.environ


def is_aml():
    return "AZ_BATCH_MASTER_NODE" in os.environ or "AZ_BATCHAI_MPI_MASTER_NODE" in os.environ


def get_filesystem():
    if is_amlk8s() or is_aml():
        fs = fsspec.filesystem("file")
        print("======Using native file system======")
        return fs
    else:
        abfs = AzureBlobFileSystem(connection_string=os.environ["AZURE_BLOB_CONNECTION_STRING"])
        print("======Using azure blob file system======")
        return abfs


class FileSystem:
    def __init__(self):
        if is_amlk8s() or is_aml():
            print("======Using native file system======")
            self.fs = "native"
        else:
            print("======Using azure blob file system======")
            self.fs = "blob"
            self.client = BlobServiceClient.from_connection_string(os.environ["AZURE_BLOB_CONNECTION_STRING"])

    def open(self, path):
        if self.fs == "native":
            return open(path)
        else:
            container = path.split("/")[0]
            blob = path[len(container) + 1 :]
            blob_client = self.client.get_blob_client(container=container, blob=blob)
            data = blob_client.download_blob()
            return io.TextIOWrapper(io.BytesIO(data.readall()), encoding="utf-8")

    def exists(self, path):
        if self.fs == "native":
            return osp.exists(path)
        else:
            container = path.split("/")[0]
            blob = path[len(container) + 1 :]
            blob_client = self.client.get_blob_client(container=container, blob=blob)
            return blob_client.exists()


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
