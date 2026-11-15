#!/usr/bin/env python3
"""tools/libcuda_check.py's network guard: redirects stay on HTTPS, and the GitHub token is sent to its own host
only, never to the host a redirect points at. No network: the redirect handler is called as urllib would."""
import importlib.util
import os
import sys
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "libcuda_check.py"
spec = importlib.util.spec_from_file_location("libcuda_check", TOOL)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

API = "https://api.github.com/repos/NVIDIA/open-gpu-kernel-modules/tags?per_page=100&page=1"


def redirect(url, newurl, code=302):
    """What urllib does on a redirect response: ask the opener's handler for the follow-up request."""
    handler = next(h for h in check.OPENER.handlers if isinstance(h, check.SafeRedirects))
    return handler.redirect_request(check.request(url), None, code, "Found", {}, newurl)


class Redirects(unittest.TestCase):
    def setUp(self):
        self.saved = os.environ.get("GITHUB_TOKEN")
        os.environ["GITHUB_TOKEN"] = "secret-token"

    def tearDown(self):
        if self.saved is None:
            os.environ.pop("GITHUB_TOKEN", None)
        else:
            os.environ["GITHUB_TOKEN"] = self.saved

    def test_token_only_for_the_api(self):
        self.assertEqual(check.request(API).get_header("Authorization"), "Bearer secret-token")
        self.assertIsNone(check.request("https://download.nvidia.com/XFree86/").get_header("Authorization"))

    def test_same_host_keeps_the_token(self):
        new = redirect(API, "https://api.github.com/repositories/1/tags?page=1")
        self.assertEqual(new.get_header("Authorization"), "Bearer secret-token")

    def test_other_host_drops_the_token(self):
        for code in (301, 302, 303, 307, 308):
            new = redirect(API, "https://codeload.github.com/elsewhere", code)
            self.assertIsNone(new.get_header("Authorization"), code)
            self.assertEqual(new.get_header("User-agent"), check.USER_AGENT)

    def test_other_port_drops_the_token(self):
        self.assertIsNone(redirect(API, "https://api.github.com:8443/x").get_header("Authorization"))

    def test_non_https_redirect_refused(self):
        for newurl in ("http://api.github.com/x", "ftp://example.com/x", "file:///etc/passwd"):
            with self.assertRaises(check.CheckError, msg=newurl):
                redirect("https://download.nvidia.com/XFree86/", newurl)

    def test_non_https_request_refused(self):
        with self.assertRaises(check.CheckError):
            check.request("http://download.nvidia.com/XFree86/")

    def test_every_download_goes_through_the_guarded_opener(self):
        source = TOOL.read_text()
        self.assertNotIn("urllib.request.urlopen(", source)
        self.assertGreaterEqual(source.count("OPENER.open("), 3)


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False).result.wasSuccessful() else 1)
