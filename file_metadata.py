from dataclasses import dataclass, field
from typing import Dict, List, Optional
from llama_index.core import Document
from pydantic import BaseModel, Field
from datetime import datetime
import os
from pathlib import Path
from stat import *
import mimetypes

class FileMetadata(BaseModel):
    """File system metadata using Python's built-in file information."""
    # Basic file information
    doc_id: str
    filename: str
    file_path: str
    file_size: int

    # Time information
    created_time: datetime
    modified_time: datetime
    accessed_time: datetime
    
    # File type information
    file_type: str  # MIME type
    file_extension: str
    
    # File permissions and ownership
    permissions: int  # Unix-style permissions
    owner_uid: int
    group_gid: int
    
    # Additional LlamaIndex metadata
    custom_metadata: Dict = Field(default_factory=dict)

    @classmethod
    def from_file_path(cls, file_path, doc_id: Optional[str] = None) -> "FileMetadata":
        """Create FileMetadata instance from a file path."""
        path = Path(file_path)
        stat_info = path.stat()
        
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(str(path))
        
        return cls(
            doc_id=doc_id or str(path.absolute()),
            filename=path.name,
            file_path=str(path.absolute()),
            file_size=stat_info.st_size,
            created_time=datetime.fromtimestamp(stat_info.st_ctime),
            modified_time=datetime.fromtimestamp(stat_info.st_mtime),
            accessed_time=datetime.fromtimestamp(stat_info.st_atime),
            file_type=mime_type or "application/octet-stream",
            file_extension=path.suffix,
            permissions=stat_info.st_mode,
            owner_uid=stat_info.st_uid,
            group_gid=stat_info.st_gid
        )
