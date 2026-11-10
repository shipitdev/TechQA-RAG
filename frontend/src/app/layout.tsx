import type { Metadata } from "next";
import localFont from "next/font/local";

const geist = localFont({ src: "./fonts/geist-latin.woff2", variable: "--font-geist", display: "swap" });
const mono = localFont({ src: "./fonts/geist-mono-latin.woff2", variable: "--font-geist-mono", display: "swap" });
import "./globals.css";
import "./genui.css";

export const metadata: Metadata = {
  title: "Fieldnote — Research workspace",
  description: "Search a local technical knowledge base and inspect the verified sources behind each answer.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${geist.variable} ${mono.variable}`}>{children}</body>
    </html>
  );
}
