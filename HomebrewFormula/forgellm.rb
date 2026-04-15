class Forgellm < Formula
  desc "Rust-native AOT ML compiler for small LLMs (1M-7B parameters)"
  homepage "https://forgellm.dev"
  version "0.6.1"
  license "MIT"

  on_macos do
    on_arm do
      url "https://github.com/sauravpanda/forge-llm/releases/download/v0.6.1/forge-v0.6.1-aarch64-apple-darwin.tar.gz"
      sha256 "cbcbb1e64012ea55d0dfa50a285a3c59efa4cea0edd787807f8e5c6c49da70a4"
    end
  end

  def install
    bin.install "forge"
    doc.install "README.md", "CHANGELOG.md", "LICENSE"
  end

  test do
    assert_match "forge #{version}", shell_output("#{bin}/forge --version")
    assert_match "Architectures:", shell_output("#{bin}/forge --version")
  end
end
