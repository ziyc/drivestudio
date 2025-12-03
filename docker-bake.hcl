variable "TAG_PREFIX" {
  default = "drivestudio"
}

group "default" {
  targets = ["base", "nuscenes"]
}

target "base" {
  context    = "."
  dockerfile = "docker/Dockerfile.base"
  tags       = ["${TAG_PREFIX}:base"]
  output     = ["type=docker"]
}

target "nuscenes" {
  context    = "."
  dockerfile = "docker/Dockerfile.nuscenes"
  tags       = ["${TAG_PREFIX}:nuscenes"]
  output     = ["type=docker"]
  args = {
    BASE_IMAGE = "base"
  }
  contexts = {
    base = "target:base"
  }
  depends_on = ["base"]
}
