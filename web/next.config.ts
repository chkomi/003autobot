import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,

  // AUTOBOT_API_URL 이 Vercel 환경변수로 주입됨
  // 서버 컴포넌트에서만 사용 (브라우저에 노출되지 않음)
  env: {
    AUTOBOT_API_URL: process.env.AUTOBOT_API_URL ?? "",
    AUTOBOT_API_KEY: process.env.AUTOBOT_API_KEY ?? "",
  },
};

export default nextConfig;
