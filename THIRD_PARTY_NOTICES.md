# Third-party notices

SCW-Pose uses **Ultralytics 8.4.188**, including its YOLO26 pose model,
preprocessing, prediction utilities, and result containers. Ultralytics is
licensed under the **GNU Affero General Public License, version 3 (AGPL-3.0)**.
This release is distributed under AGPL-3.0; the complete license text is included
in [LICENSE](LICENSE).

- Upstream source: https://github.com/ultralytics/ultralytics
- Upstream licensing: https://www.ultralytics.com/license

The SCW-Pose refinement modules and inference/evaluation entry points extend the
upstream model. They do not replace the upstream authors' ownership or license
notices. The Ultralytics package itself is installed as a dependency, not vendored
in this repository.

PyTorch, Torchvision, OpenCV, NumPy, and other packages listed in the dependency
files are also installed dependencies, not vendored source. They retain their
respective upstream licenses and notices, supplied with their distributions.
