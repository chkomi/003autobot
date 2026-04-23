import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "003 Autobot",
  description: "OKX long/short opportunity intelligence and backtesting workspace.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
