// Package assets resolves named external data dependencies (model weights,
// vocab files, demo corpora) to local on-disk paths, downloading on first
// use and SHA-verifying every byte before returning.
package assets

// Asset describes one downloadable resource by content hash.
//
// URL is the canonical fetch source. SHA256 is the lowercase hex SHA-256 of
// the expected bytes (verified after download and on every cache hit). Size
// is the expected size in bytes, or -1 when unknown; when positive it is
// used for an up-front disk-space precheck.
type Asset struct {
	Name   string
	URL    string
	SHA256 string
	Size   int64
}

// Registry is the canonical table of assets the project knows how to fetch.
// Adding a new asset means adding an entry here with its pinned SHA.
var Registry = map[string]Asset{
	"shakespeare": {
		Name:   "shakespeare",
		URL:    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
		SHA256: "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed",
		Size:   1115394,
	},
	"gpt2-safetensors": {
		Name:   "gpt2-safetensors",
		URL:    "https://huggingface.co/gpt2/resolve/main/model.safetensors",
		SHA256: "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707",
		Size:   548105171,
	},
	"gpt2-vocab": {
		Name:   "gpt2-vocab",
		URL:    "https://huggingface.co/gpt2/resolve/main/vocab.json",
		SHA256: "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
		Size:   1042301,
	},
	"gpt2-merges": {
		Name:   "gpt2-merges",
		URL:    "https://huggingface.co/gpt2/resolve/main/merges.txt",
		SHA256: "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
		Size:   456318,
	},
	// cifar10-binary is the official Krizhevsky 2009 binary distribution
	// (cifar-10-binary.tar.gz). The pinned SHA-256 matches the canonical
	// upstream tarball; if the first download fails verification with a
	// fresh hash, confirm against the upstream README before bumping.
	"cifar10-binary": {
		Name:   "cifar10-binary",
		URL:    "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
		SHA256: "c4a38c50a1bc5f3a1c5537f2155ab9d68f9f25eb1ed8d9ddda3db29a59bca1dd",
		Size:   170052171,
	},
}
