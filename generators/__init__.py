"""Generator sub-package.

Importing this package registers all built-in pattern generators into the
shared PATTERNS registry. Add a new module here and import it below to make
its patterns visible to the CLI automatically.
"""

from generators import baseline, counting, dyck, structural
